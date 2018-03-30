#include <iostream>
using namespace std;
template <typename T>

void bubble_sort(T lst[],int len)
{
	int i,j;
	T temp;
	for (i=0;i<len-1;i++)
	{
		for (j=0;j<len-1;j++)
		{
			if (lst[j]>lst[j+1])
			{
				swap(lst[j],lst[j+1]);
			}
		}
	}
}

template <typename T>//在每个使用到T类型的函数前都需要加template 
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
	bubble_sort(lst,len);
	for (int i=0;i<len;i++)
	{
		cout << lst[i] << " ";
	}
	cout << endl;
	return 0;
}
