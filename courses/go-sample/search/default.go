package search

// defaultMatcher实现了默认匹配器
/*
空结构在创建实例时，不会分配任何内存。这种结构很适合创建没有任何状态的类型。
对于默认匹配器来说，不需要维护任何状态，所以我们只要实现对应的接口就行。
*/
type defaultMatcher struct{} //使用一个空结构声明了一个名叫 defaultMatcher 的结构类型。

// init函数将默认匹配器注册到程序里
func init() {
	var matcher defaultMatcher
	Register("default", matcher)
}

// Search 实现了默认匹配器的行为
/*
Search 方法的声明也声明了 defaultMatcher 类型的值的接收者
如果声明函数的时候带有接收者，则意味着声明了一个方法。这个方法会和指定的接收者的类型绑在一起。
在我们的例子里，Search 方法与 defaultMatcher 类型的值绑在一起。
这意味着我们可以使用 defaultMatcher 类型的值或者指向这个类型值的指针来调用 Search 方法。无论我们是使用接收者类型的值来调用这个方，
还是使用接收者类型值的指针来调用这个方法，编译器都会正确地引用或者解引用对应的值，作为接收者传递给 Search 方法
因为大部分方法在被调用后都需要维护接收者的值的状态，所以，一个最佳实践是，将方法的接收者声明为指针。
对于 defaultMatcher 类型来说，使用值作为接收者是因为创建一个 defaultMatcher 类型的值不需要分配内存。
由于 defaultMatcher 不需要维护状态，所以不需要指针形式的接收者。
除了 Search 方法，defaultMatcher 类型不需要为实现接口做更多的事情了。
从这段代码之后，不论是 defaultMatcher 类型的值还是指针，都满足 Matcher 接口，都可以作为 Matcher 类型的值使用。（关键）
defaultMatcher 类型的值和指针现在还可以作为 Matcher 的值，赋值或者传递给接受 Matcher 类型值的函数。
*/
func (m defaultMatcher) Search(feed *Feed, searchTerm string) ([]*Result, error) { //defaultMatcher 类型实现 Matcher 接口
	return nil, nil //实现接口的方法 Search 只返回两个 nil 值
}
