#include <curses.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/time.h>

#define CHAR_BIRD 'O'  // 定义 bird 字符
#define CHAR_STONE '*'  // 定义组成柱子的石头
#define CHAR_BLANK ' '  // 定义空字符

typedef struct node{  //背景中的柱子用单向链表存储
	int x, y;
	struct node *next;
}node, *Node;

Node head, tail;
int bird_x, bird_y;
int ticker;

void init();  // 初始化函数，统筹游戏各项的初始化工作
void init_bird();  // 初始化 bird 位置坐标
void init_draw();  // 初始化背景
void init_wall();  // 初始化存放柱子的链表的链表头
void init_head();  // 初始化存放柱子的链表
void drop(int sig);  // 信号接收函数，用来接收到系统信号，从右向左移动柱子
int set_ticker(int n);  // 设置内核的定时周期

int main()
{
	char ch;
    
    init();
    while(1) 
    {
    	ch = getch();  // 获取键盘输入
    	if(ch == ' ' || ch == 'w' || ch == 'W')  // 按下空格或者 W 时
    	{
    		move(bird_y, bird_x); // 移动 bird ，并进行重绘
    		addch(CHAR_BLANK);
    		refresh();
    		bird_y--;
    		move(bird_y, bird_x);
    		addch(CHAR_BIRD);
    		refresh();
    		if((char)inch() == CHAR_STONE)  // 如果 bird 撞到了柱子，结束游戏
    		{
    			set_ticker(0);
    			sleep(1);
    			endwin();
    			exit(0);
    		}
    	}
    	else if(ch == 'z' || ch == 'Z')  // 暂停
    	{
    		set_ticker(0);
    		do 
    		{
    			ch = getch();
    		} while(ch != 'z' && ch != 'Z');
    		set_ticker(ticker);
    	}
    	else if(ch == 'q' || ch == 'Q')  // 退出
    	{
    		sleep(1);
    		endwin();
    		exit(0);
    	}
    }
    return 0;
}

int set_ticker(int n_msec)
{
	struct itimerval timeset;
	long n_sec, n_usec;

	n_sec = n_msec / 1000;
	n_usec = (n_msec % 1000) * 1000L;

	timeset.it_interval.tv_sec = n_sec;
	timeset.it_interval.tv_usec = n_usec;

	timeset.it_value.tv_sec = n_sec;
	timeset.it_value.tv_usec = n_usec;

	return setitimer(ITIMER_REAL, &timeset, NULL);
}

void drop(int sig)
{
    int j; 
    Node tmp, p;
    // 将原先 bird 位置的符号清除
    move(bird_y, bird_x);
    addch(CHAR_BLANK);
    refresh();
     // 更新 bird 的位置并刷新屏幕
    bird_y++;
    move(bird_y, bird_x);
    addch(CHAR_BIRD);
    refresh();
    // 如果撞上柱子则结束游戏
    if((char)inch() == CHAR_STONE) 
    {
        set_ticker(0);
    	sleep(1);
    	endwin();
    	exit(0);
    }
    // 检测第一块墙是否超出边界
    p = head->next;
    if(p->x < 0) 
    {
    	head->next = p->next;
    	free(p);
    	tmp = (node *)malloc(sizeof(node));
    	tmp->x = 99;
    	do 
    	{
    		tmp->y = rand() % 16;
    	} while(tmp->y < 5);
    	tail->next = tmp;
    	tmp->next = NULL;
    	tail = tmp;
    	ticker -= 10;
    	set_ticker(ticker);
    }
	 // 绘制新的柱子
    for(p = head->next; p->next != NULL; p->x--, p = p->next) 
    {
		// 使用 CHAR_BLANK 替代原先的 CHAR_STONE
    	for(j = 0; j < p->y; j++) 
    	{
    		move(j, p->x);
    		addch(CHAR_BLANK);
    		refresh();
    	}
    	for(j = p->y+5; j <= 23; j++) 
    	{
    		move(j, p->x);
    		addch(CHAR_BLANK);
    		refresh();
    	}
    
    	if(p->x-10 >= 0 && p->x < 80) 
    	{
    		for(j = 0; j < p->y; j++) 
    		{
    			move(j, p->x-10);
    			addch(CHAR_STONE);
    			refresh();
    		}
    		for(j = p->y + 5; j <= 23; j++) 
    		{
    			move(j, p->x-10);
    			addch(CHAR_STONE);
    			refresh();
    		}
    	}
    }
    tail->x--;
}

void init()
{
	initscr();
	cbreak();
	noecho();
	curs_set(0);
	srand(time(0));
	signal(SIGALRM, drop);

	init_bird();
	init_head();
	init_wall();
	init_draw();
	sleep(1);
	ticker = 500;
	set_ticker(ticker);
}

void init_bird()
{
	bird_x = 5;
	bird_y = 15;
	move(bird_y, bird_x);
	addch(CHAR_BIRD);
	refresh();
	sleep(1);
}

void init_head()
{
	Node tmp;

	tmp = (node *)malloc(sizeof(node));
	tmp->next = NULL;
	head = tmp;
	tail = head;
}

void init_wall()
{
	int i;
	Node tmp, p;

	p = head;
	for(i = 19; i <= 99; i += 20)
	{
		tmp = (node *)malloc(sizeof(node));
		tmp->x = i;
		do{
			tmp->y = rand() % 16;
		}while(tmp->y < 5);
		p->next = tmp;
		tmp->next = NULL;
		p = tmp;
	}
	tail = p;
}

void init_draw()
{
	Node p;
	int i, j;

	for(p = head->next; p->next != NULL; p = p->next)
	{
		for(i = p->x; i > p->x-10; i--)
		{
			for(j = 0; j < p->y; j++)
			{
				move(j, i);
				addch(CHAR_STONE);
				refresh();
			}
			for(j = p->y+5; j <= 23; j++)
			{
				move(j, i);
				addch(CHAR_STONE);
				refresh();
			}
		}
		sleep(1);
	}
}