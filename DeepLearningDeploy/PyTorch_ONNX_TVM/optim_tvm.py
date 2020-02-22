# load tvm model and run
import numpy as np 
import time
import os
import torch
import onnx

import tvm
from tvm.contrib import graph_runtime
from tvm.runtime.module import load_module
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner

# target = "llvm -mcpu=Ivy Bridge"
# target = "llvm"
target = "llvm -mcpu=haswell"
batch_size = 1
dtype = "float32"
model_name = "east"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name
input_name = "input.1"
num_thread = 2
os.environ["TVM_NUM_THREADS"] = str(num_thread)

tuning_option = {
    'log_filename': log_file,
    'tuner': 'random',
    'early_stopping': 1,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1,
                                   min_repeat_ms=1000),
    ),
}

def load():

    model_path = "./east.so"
    params_path = "./params.params"
    graph_path = "./graph.json"

    # load model
    # module = tvm.module.load(model_path)
    module = load_module(model_path)
    graph = open(graph_path,'r').read()
    params = bytearray(open(params_path,'rb').read())

    return module, graph, params

def run_raw(clock = False):
    module, graph, params = load()
    dshape = (1,3,256,256)
    dtype = "float32"
    ctx = tvm.cpu(0)
    dummpy_input = np.ones(dshape)


    model = graph_runtime.create(graph, module, ctx)
    model.load_params(params)

    model.set_input("input.1", dummpy_input)
    model.run()
    if clock:
        ftimer = model.module.time_evaluator("run",ctx,number=10,repeat=3)
        prof_res = np.array(ftimer().results) * 1000
        print("Mean inference time (std dev) : %.2f ms (%.2f ms)" % 
            (np.mean(prof_res),np.std(prof_res)))

    score = model.get_output(0).asnumpy()
    geo = model.get_output(1).asnumpy()
    print("score shape {}\t geo shape {}\t".format(score.shape, geo.shape))



def tune_kernels(tasks, measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):
    for i,tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        op_name = tsk.workload[0]
        print("op_name",op_name)
        if op_name == "conv2d":
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key='direct')
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


def tune_graph(graph, dshape, records, opt_sch_file, use_DP = False):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)

def tune_and_evaluate(tuning_opt):
    x = torch.randn(1, 3, 256,256).data.numpy()
    data_shape = (1,3,256,256)

    model = onnx.load("./east.onnx")
    shape_dict = {input_name:x.shape}
    mod,params = relay.frontend.from_onnx(model,shape_dict)
    tasks = autotvm.task.extract_from_program(
        mod['main'],target=target,params=params,ops = (relay.op.nn.conv2d,))
    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)
    print("Saving graph...")
    tune_graph(mod['main'], data_shape, log_file, graph_opt_sch_file)

    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params
            )

        ctx = tvm.cpu()
        data_tvm = tvm.nd.array(np.random.uniform(size=data_shape).astype(dtype))
        module = graph_runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        # module.load_params(**params)
        module.set_input(**params)

        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

    # save model
    export_path = "./opt_east.so"
    lib.export_library(export_path)

    graph_json = "./opt_graph.json"
    with open(graph_json,'w') as f:
        f.write(graph)

    params_json = "./opt_params.params"
    with open(params_json,'wb') as f:
        f.write(relay.save_param_dict(params))




if __name__ == "__main__":
    import sys
    arg = sys.argv
    if len(arg) == 1:
        run_raw(clock=True)
    elif arg[1] == "tune":
        tune_and_evaluate(tuning_option)
