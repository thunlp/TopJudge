#include<cstdio>
#include<cstdlib>
#include<cstring>

using namespace std;

int file_num = 20;

char djm[1000];

FILE *f[100];

int main()
{
	freopen("/disk/mysql/mysql/Law1/out.txt","r",stdin);

	for (int a=0;a<file_num;a++)
	{
		sprintf(djm,"/disk/mysql/law_data/origin_split/%d",a);
		f[a] = fopen(djm,"w");
	}

	int cnt=0,line=0;
	char c;
	while (~scanf("%c",&c))
	{
		fprintf(f[cnt],"%c",c);
		if (c=='\n')
		{
			line++;
			if (line % 100000 == 0) 
			{
				printf("%d\n",line);
				cnt++;
				if (cnt==file_num) cnt=0;
			}
		}
	}

	return 0;
}
