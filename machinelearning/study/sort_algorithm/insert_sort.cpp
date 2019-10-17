#include <iostream>
using namespace std;

template <typename T>
void insert_sort(T lst[],int len)
{
	int i,j;
	T temp;
	for(i=0;i<=len-1;i++)
	{
		temp = lst[i];
		j = i - 1;
		while(j>=0&&lst[j]>temp)
		{
			lst[j+1] = lst[j];
			j--;
		}
		lst[j+1] = temp;
	}
} 

int main()
{
	float lst[] = {11,7,6,54,4,3,32,22,1,2,3,4,5,6,7};
	int len = (int) sizeof(lst)/sizeof(*lst);
	insert_sort(lst,len);
	for (int i=0;i<len;i++)
	{
		cout << lst[i] << " ";
	}
	cout << endl;
	return 0;
}
