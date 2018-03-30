#include <iostream>
using namespace std;

template <typename T>
void select_sort(T lst[],int len)
{
	int i,j;
	for(i=0;i<=len-1;i++)
	{
		int min_index = i;
		for(j=i;j<=len-1;j++)
		{
			if(lst[j]<lst[min_index])
			{
				min_index = j;
			}
		}
		swap(lst[i],lst[min_index]);
	}
}

template <typename T>
void swap(T x, T y)
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
	select_sort(lst,len);
	for (int i=0;i<len;i++)
	{
		cout << lst[i] << " ";
	}
	cout << endl;
	return 0;
}

