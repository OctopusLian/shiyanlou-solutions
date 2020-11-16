#include <stdio.h>
#include <stdlib.h>

#define N	15

//定义一个数组并为每一个元素赋初值0
int chessboard[N + 1][N + 1] = { 0 };

//用来记录轮到玩家1还是玩家2，奇数表示轮到玩家1，偶数轮到玩家2
int whoseTurn = 0;

void initGame(void);
void printChessboard(void);
void playChess(void);
int judge(int, int);

int main(void)
{
	//自定义函数，用来初始化游戏，也就是显示欢迎界面并且进入游戏显示棋盘
	initGame();
	//这个循环就是用来让两个玩家轮流落子的
	while (1)
	{
		//每次循环自增1，这样就可以做到2人轮流
		whoseTurn++;
		//自定义函数，用来执行落子操作
		playChess();
	}

	return 0;
}

/*
在这个函数中，我们要实现的功能是
 * 显示一个简单的欢迎界面
 * 要求输入Y之后显示出棋盘
*/
void initGame(void)
{
	char c;

	printf("欢迎^_^请输入y进入游戏");
	c = getchar();
	if ('y' != c && 'Y' != c)
		exit(0);

	//清屏,windows下为system("cls")
	system("clear");
	//这里我们又调用了一个自定义函数，函数的功能是打印出棋盘
	printChessboard();
}

/*
功能：
* 打印出行号和列号，并打印出棋盘
* 数组元素的值为0，打印出星号（*），表示该位置没有人落子
* 数组元素的值为1，打印实心圆（●，玩家1的棋子）
* 数组元素的值为2，打印空心圆（○，玩家2的棋子）
*/
void printChessboard(void)
{
	int i, j;

	for (i = 0; i <= N; i++)
	{
		for (j = 0; j <= N; j++)
		{
			if (0 == i)  //这样可以打印出列号
				printf("%3d", j);
			else if (j == 0)  //打印出行号
				printf("%3d", i);
			else if (1 == chessboard[i][j])
			//windows下●占2列，前面只需加一个空格
				printf("  ●");
			else if (2 == chessboard[i][j])
				printf("  ○");
			else
				printf("  *");
		}
		printf("\n");
	}
}

/*
函数功能：
* 要求玩家输入准备落子的位置
* 如果当前是玩家1落子，就将1赋值给数组中对应位置的元素
* 如果当前是玩家2落子，就将2赋值给数组中对应位置的元素
* 每次落子完毕，判断当前玩家是否获胜
*/
void playChess(void)
{
	int i, j, winner;
	//判断轮到玩家1还是玩家2，然后把值赋给数组中对应的元素
	if (1 == whoseTurn % 2)
	{
		printf("轮到玩家1，请输入棋子的位置，格式为行号+空格+列号：");
		scanf("%d %d", &i, &j);
		
		//修复 #issue1 Bug
		while(chessboard[i][j] != 0)
		{
			printf("您要下的位置已经被占用了哦，请重新输入："); 
			scanf("%d %d", &i, &j);
		}
		
		chessboard[i][j] = 1;
	}
	else
	{
		printf("轮到玩家2，请输入棋子的位置，格式为行号+空格+列号：");
		scanf("%d %d", &i, &j);
		
		//修复 #issue1 Bug
		while(chessboard[i][j] != 0)
		{
			printf("您要下的位置已经被占用了哦，请重新输入："); 
			scanf("%d %d", &i, &j);
		}
		
		chessboard[i][j] = 2;
	}
	//重新打印一次棋盘
	system("clear");
	//再次调用了这个函数
	printChessboard();
	
	//自定义函数judge，判断当前玩家下完这步棋后，他有没有获胜
	if (judge(i, j))
	{
		if (1 == whoseTurn % 2)
		{
			printf("玩家1胜！\n");
			exit(0);	//修复 #issue2 Bug
		}
		else
		{
			printf("玩家2胜！\n");
			exit(0);	//修复 #issue2 Bug
		}
	}
}

/*
函数参数：

* x：当前落子的行号
* y：当前落子的列号

返回值：

* 1或0。1表示当前玩家落子之后出现五子连一线，也就是当前玩家获胜

前两个for循环判断竖直方向上是否有五子连线出现。而后两个for循环是判断两个斜线方向上是否有五子连线出现。
*/
int judge(int x, int y)
{
	int i, j;
	int t = 2 - whoseTurn % 2;

	for (i = x - 4, j = y; i <= x; i++)
	{
		if (i >= 1 && i <= N - 4 && t == chessboard[i][j] && t == chessboard[i + 1][j] && t == chessboard[i + 2][j] && t == chessboard[i + 3][j] && t == chessboard[i + 4][j])
			return 1;
	}
	for (i = x, j = y - 4; j <= y; j++)
	{
		if (j >= 1 && j <= N - 4 && t == chessboard[i][j] && t == chessboard[i][j + 1] && t == chessboard[i][j + 1] && t == chessboard[i][j + 3] && t == chessboard[i][j + 4])
			return 1;
	}
	for (i = x - 4, j = y - 4; i <= x, j <= y; i++, j++)
	{
		if (i >= 1 && i <= N - 4 && j >= 1 && j <= N - 4 && t == chessboard[i][j] && t == chessboard[i + 1][j + 1] && t == chessboard[i + 2][j + 2] && t == chessboard[i + 3][j + 3] && t == chessboard[i + 4][j + 4])
			return 1;
	}
	for (i = x + 4, j = y - 4; i >= 1, j <= y; i--, j++)
	{
		if (i >= 1 && i <= N - 4 && j >= 1 && j <= N - 4 && t == chessboard[i][j] && t == chessboard[i - 1][j + 1] && t == chessboard[i - 2][j + 2] && t == chessboard[i - 3][j + 3] && t == chessboard[i - 4][j + 4])
			return 1;
	}

	return 0;
}