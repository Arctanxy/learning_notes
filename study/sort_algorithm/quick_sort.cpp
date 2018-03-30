#include <iostream>
using namespace std;

template <typename T>
int partition(T lst[],int low,int high)
{
	T pivot = lst[high-1];
	int small = low - 1;
	int i;
	for(i=low;i<high;i++)
	{
		if(lst[i]<pivot)
		{
			small ++;//记录最后一个小于pivot的元素的index 
			swap(lst[i],lst[small]);
		}
	}
	if(lst[small+1] > pivot)
	{
		swap(lst[small+1],lst[high-1]);
	}
	return small + 1;
}

template <typename T>
void quick_sort(T lst[],int low,int high)
{
	if(low < high)
	{
		int p = partition(lst,low,high);
		quick_sort(lst,low,p);
		quick_sort(lst,p+1,high);
	}
}

template <typename T>
void swap(T x,T y)
{
	T temp;
	temp = x;
	x = y;
	y = temp;
}

int main()
{
	float lst[] = {11,7,6,54,4,3,32,22,1,2,3,4,5,6,7};
	int len = (int) sizeof(lst)/sizeof(*lst);
	quick_sort(lst,0,len);
	for (int i=0;i<len;i++)
	{
		cout << lst[i] << " ";
	}
	cout << endl;
	return 0;
}
