#include<cstdio>
#include<cstdlib>
#include<cstring>

using namespace std;

int file_num = 20;

char djm[1000];

FILE *f[100];

int main()
{
	freopen("/data/disk1/private/guozhipeng/out1.txt","r",stdin);

	for (int a=0;a<file_num;a++)
	{
		sprintf(djm,"%d",a);
		f[a] = fopen(djm,"w");
	}

	int cnt=0,line=0;
	char c;
	while (~scanf("%c",&c))
	{
		fprintf(f[cnt],"%c",c);
		if (c=='\n')
		{
			cnt++;
			if (cnt==file_num) cnt=0;
			line++;
			if (line % 100000 == 0) printf("%d\n",line);
		}
	}

	return 0;
}
