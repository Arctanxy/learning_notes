//https://leetcode.com/problems/jewels-and-stones/description/

#include <iostream>
#include <string.h>
using namespace std;

//��򵥵�O(N^2)���Ӷ��㷨
int numJewelsInStones(char* J, char* S) {
		int num = 0;
		for (int i = 0; i < strlen(J); i++) {
			for (int j = 0; j < strlen(S); j++) {
				if (S[j] == J[i]) {
					num++;
				}
			}
		}
		return num;
}

int main()
{
	char J[50], S[50];
	cout << "input J";
	cin >> J;
	cout << "input S";
	cin >> S;
	cout << endl << numJewelsInStones(J, S);
	return 0;
}


