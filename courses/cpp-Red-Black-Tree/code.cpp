#include<stdio.h>
using namespace std;
int read()
{int s=0,f=1;char ch=getchar();
 while(ch<'0'||ch>'9'){if(ch=='-')f=-1;ch=getchar();}
 while(ch>='0'&&ch<='9'){s=(s<<1)+(s<<3)+ch-'0';ch=getchar();}
 return s*f;
}
//smile please
int n,m;
int ans;
int main()
{
    //freopen(".in","r",stdin);
    //freopen(".out","w",stdout);
    n=read();
    m=n+1;
    ans=0;
    while(m>1)
      {if(m&1)
         ans++;
       m/=2;
      }
    printf("%d\n",ans);
    m=n+1;
    ans=0;
    while(m>1)
      {if(m==2)ans++;
       if((m&3)==1)
         ans+=m/4*2-1,m/=4,m++;
       else if((m&3)==2)
         ans+=m/4*2,m/=4,m++;
       else if((m&3)==3)
         ans+=m/4*2+1,m/=4,m++;
       else
         if(!(m&3))
           ans+=m/4*2,m/=4;
      }
    printf("%d\n",ans);
    //fclose(stdin);
    //fclose(stdout);
    return 0;
}
﻿
