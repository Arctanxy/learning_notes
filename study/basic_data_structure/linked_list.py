#链表节点

class LNode:
	def __init__(self,elem,next_ = None):
		self.elem = elem
		self.next_ = next_

#链表

class LList:
	def __init__(self):
		self.head_ = None
	
	#检查是否为空表
	def is_empty(self):
		return self.head_ is None
	
	#在表头插入元素
	def prepend(self,elem):
		self.head_ = LNode(elem,self.head_)
	
	#删除并返回表头数据
	def pop(self):
		if self.head_ is None:
			raise LinkedListUnderFlow('in pop')
		e = self.head_.elem
		self.head_ = self.head_.next_
		return e

	#后端插入
	def append(self,elem):
		if self.head_ is None:
			self.head_ = LNode(elem)
			return #如果头节点为空则赋值给头节点并提前结束
		p = self.head_
		while p.next_ is not None:
			p = p.next_
		p.next_ = LNode(elem)
	
	#删除并返回表末数据
	def pop_last(self):
		if self.head_ is None:
			raise LinkedListUnderFlow("in pop_last")
		p = self.head_
		if p.next_ is None:#如果只有一个元素则操作于pop()一致
			e = p.elem
			self.head_ = None
			return e
		while p.next_.next_ is not None:
			#遍历，直到找到p.next_.next_为None的元素，此时p.next_即为最后一个元素
			p = p.next_
		e = p.next_.elem
		p.next_ = None
		return e
	
	#寻找满足条件的元素,pred为筛选函数
	def find(self,pred):
		p = self.head_
		while p is not None:
			if pred(p.elem):
				return p.elem
			p = p.next_
	
	#对链表元素批量操作,proc为操作函数
	def for_each(self,proc):
		p = self.head_
		while p is not None:
			proc(p.elem)
			p = p.next_
	
	#返回链表中所有元素（以生成器的形式）
	def elements(self):
		p = self.head_
		while p is not None:
			yield p.elem
			p = p.next_
	
	#寻找多个满足条件的元素（以生成器的形式）
	def filter(self,pred):
		p = self.head_
		while p is not None:
			if pred(p.elem):
				yield p.elem
			p = p.next_
