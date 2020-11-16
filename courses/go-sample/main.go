package main

import (
	"log"
	"os"

	_ "./matchers" //这个技术是为了让 Go 语言对包做初始化操作，但是并不使用包里的标识符
	"./search"
)

// init在main之前调用
func init() {
	// 将日志输出到标准输出
	log.SetOutput(os.Stdout)
}

func main() {
	// 使用特定的项做搜索
	search.Run("president")
}
