#编译方式
CC = g++ 
#C++版本，显示所有警告
VERSION = -g -std=c++11 -Wall
#头文件和库文件
INCLUDE = -I../include -lglut -lGL
#目标文件，最后生成文件
TARGET = ../bin/Pacman
#源代码路径
SRCS = $(wildcard *.cpp) 
#编译为.o文件
OBJS = $(patsubst %cpp,%o,$(SRCS))

all:$(TARGET)
#执行编译    
$(TARGET):$(OBJS)
    $(CC) $(OBJS) $(VERSION) $(INCLUDE) -o $(TARGET)

%.o:%.cpp
    $(CC) $(VERSION) $(INCLUDE) -c $<
#清除
.PHONY:clean
clean:
    rm -rf $(OBJS) $(TARGET)