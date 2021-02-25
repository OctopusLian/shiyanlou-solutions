# 基于GO语言Revel框架和mgo的博客  

本实验属于[实验楼](https://www.shiyanlou.com)课程《基于Reve和mgo的博客》,该课程基于[joveth](http://blog.csdn.net/joveth)的教程[Go web开发之revel+mgo](http://blog.csdn.net/jov123/article/category/2216585)改编而来。具体来说，在[joveth](http://blog.csdn.net/joveth)教程的基础上，实验楼修改了教程以适配最新版本的`Revel`。  

## 环境安装  

本课程中用到的各软件版本如下：  

```
go：1.4.3
Revel: v0.12.0
mgo:r2015.10.05 
```

在实验楼提供的实验环境中已经按照配置好全部软件包。下面我们简单描述，如何进行按照配置。如有需要，你可以参考以下教程，在本地环境中安装实验环境。  

首先，需要创建工作目录，同时设置 `GOPATH` 环境变量：  

```
$ cd /home/shiyanlou/
$ mkdir revel_mgo_blog
$ cd revel_mgo_blog
$ export GOPATH=`pwd`
```

以上步骤中，我们创建了 `/home/shiyanlou/revel_mgo_blog`目录，并将它设置为 `GOPATH`, 也是后面我们每一个实验的工作目录。结着我们安装 [Revel](http://revel.github.io/tutorial/gettingstarted.html) 和 [mgo](https://labix.org/mgo)：  

```
$ cd ~/revel_mgo_blog
$ go get github.com/revel/revel
$ go get ithub.com/revel/cmd/revel
$ go get gopkg.in/mgo.v2
```

执行完以上命令后， `Revel`和`mgo`就安装完成啦。同时我们可以看到在 `$GOPATH/bin/`也就是`/home/shiyanlou/revel_mgo_blog/bin`目录下已经有`revel`命令了。我们可以方便的使用`revel`命令，我们需要执行一下命令：  

```
$ export PATH=$PATH:$GOPATH/bin
```


现在我们可以直接执行`revel`命令了。到这里整个环境就安装配置完成。在实验楼学习时，不需要操作以上步骤，因为我们已经为你安装好了实验环境。你每次进入实验时，都需要执行以下命令保证环境变量的正确：  

```
$ cd /home/shiyanlou/revel_mgo_blog
$ export GOPATH=`pwd`
$ export PATH=$PATH:$GOPATH/bin
```


## 目录结构  

```
$ cd ~/revel_mgo_blog 
$ tree                                                                                
.
|-- README.md
|-- app
|   |-- controllers
|   |   `-- app.go
|   |-- init.go
|   |-- routes
|   |   `-- routes.go
|   |-- tmp
|   |   `-- main.go
|   `-- views
|       |-- App
|       |   `-- Index.html
|       |-- debug.html
|       |-- errors
|       |   |-- 404.html
|       |   `-- 500.html
|       |-- flash.html
|       |-- footer.html
|       `-- header.html
|-- conf
|   |-- app.conf
|   `-- routes
|-- messages
|   `-- sample.en
|-- public
|   |-- css
|   |   `-- bootstrap.css
|   |-- img
|   |   |-- favicon.png
|   |   |-- glyphicons-halflings-white.png
|   |   `-- glyphicons-halflings.png
|   `-- js
|       `-- jquery-1.9.1.min.js
`-- tests
    `-- apptest.go

14 directories, 21 files
```

+ `app`目录下面是我们的主要业务逻辑，又分为`controllers`、`views`等文件夹。`controllers`相当于我们的action主要处理都放在这里面，`views`目录存放html页面模板。`init.go`是一些初始化加载的东西。  

+ `conf`目录下的`app.conf`是应用配置文件，比如配置应用监听的端口等，`routes`文件则是我们后面需要用到的路由配置文件。  

+ `messages` 目录用于国际化处理。  

+ `public`目录存放一些静态资源，如css，image，js等  

+ `tests` 目录用于存放测试用例。  

好了，到目前为止，你应该了解`Revel`框架啦，更多信息可以参考: http://revel.github.io/index.html 。  


## Blog的首页与投稿设计  

## 首页设计

在开始设计之前，我们需要首先创建我们的博客应用：  

```
revel new GBlog
```

`Revel`默认情况下使用的是 `bootstrap 2`，现在我们将其替换成v3版本：  

```
$ cd /home/shiyanlou/revel_mgo_blog/src/GBlog/public/css
$ cp ~/revel_mgo_blog/src/git.shiyanlou.com/shiyanlou/GBlog/public/css/bootstrap.min.css ./
```

接着我们设计自己的样式，在 `/home/shiyanlou/revel_mgo_blog/src/GBlog/public/css` 中新建一个文件 `styles.css`，输入以下内容：  

```
  body{
    margin: 0 auto;
    padding: 0;
    font: 14px "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif;
    line-height: 20px;
    letter-spacing: 0.02em;
    color: #666;
    background-attachment:fixed;
}
a{
    color: #1abc9c;
    text-decoration: none;
    -webkit-transition: 0.25s;
    -moz-transition: 0.25s;
    -o-transition: 0.25s;
    transition: 0.25s;
    -webkit-backface-visibility: hidden;
}
.main-nav{
    margin: 0 auto;
    width: 692px;
    padding:0;
}
.top-bar{
    width:100%;
    background: #34495e;
    border-bottom-right-radius: 6px;
    border-bottom-left-radius: 6px;
    box-shadow:  0 2px rgba(0,0,0,0.075),0 0 6px #7aba7b;
    -webkit-box-shadow:0 2px  rgba(0,0,0,0.075),0 0 6px #7aba7b;
    -moz-box-shadow:0 2px  rgba(0,0,0,0.075),0 0 6px #7aba7b;
    margin-bottom:28px;
}
.top-bar-inner{
    min-height: 48px;
    padding:0 4px;
}
.ul-nav{
    position: relative;
    left: 0;
    display: block;
    float: left;
    margin: 0 10px 0 0;
    list-style: none;
    font-size: 18px;
    padding:0;
}
.ul-nav>li {
    position: relative;
    float: left;
    line-height: 20px;
}
.ul-nav>li>a{
    padding: 14px 24px 17px;
    text-decoration: none;
    display: block;
    color: white;
    text-shadow: 0 -1px 0 rgba(0,0,0,0.25);
}
.ul-nav>li>a:hover,.ul-nav>li>a:focus{
    color: #1abc9c;
}
.navbar-news {
    background-color: #e74c3c;
    border-radius: 30px;
    color: white;
    display: block;
    font-size: 12px;
    font-weight: 500;
    line-height: 18px;
    min-width: 8px;
    padding: 0 5px;
    position: absolute;
    right: -7px;
    text-align: center;
    text-shadow: none;
    top: 8px;
    z-index: 10;
}
.ul-nav .active > a, .ul-nav .active > a:hover, .ul-nav .active > a:focus {
  background-color: transparent;
  color: #1abc9c;
  -webkit-box-shadow: none;
  -moz-box-shadow: none;
  box-shadow: none; 
}
.cell{
    background-color:#1bc6a5;
    color: #cff3ec;
    font-size: 15px;
    border-radius: 4px;
    -webkit-border-radius: 4px;
    -moz-border-radius: 4px;
    -o-border-radius: 4px;
    -khtml-border-radius: 4px;
    padding: 18px 20px 0px 20px; 
    margin-bottom: 30px;
    box-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);

}
.cell-subject{
    margin: 0;
}
.cell-subject-title{
    color: #34495e;
    font-size: 24px;
    font-weight: 700;
    text-decoration: none;
}
a.cell-subject-title:hover{
    text-decoration: underline;
}
.subject-infor{
    color:#34495e;
    line-height: 19px;
    padding: 2px 0;
    font-size: 13px;
    margin:2px 0;
}
.cell-text{
    padding: 4px 0;
    word-break: break-all;
}
.comment-num{
    float:right;
    border: 5px solid #d7dce0;
    border-radius: 50px;
    font-size: 14px;
    line-height: 16px;
    padding: 0 4px;
    -webkit-transition: background 0.2s ease-out, border-color 0s ease-out, color 0.2s ease-out;
    -moz-transition: background 0.2s ease-out, border-color 0s ease-out, color 0.2s ease-out;
    -o-transition: background 0.2s ease-out, border-color 0s ease-out, color 0.2s ease-out;
    transition: background 0.2s ease-out, border-color 0s ease-out, color 0.2s ease-out;
    -webkit-backface-visibility: hidden;
    background-color: white;
    border-color: white;
    border-width: 2px;
    color: #BBB6B6;
}
```

接着，我们让HTML文件包含我们的`styles.css`，编辑`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/header.html`, 写入一些内容：  

```
 <!DOCTYPE html>
   <html>
    <head>
      <title>{{.title}}</title>
      <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
      <link rel="stylesheet" type="text/css" href="/public/css/bootstrap.min.css">
      <link rel="stylesheet" type="text/css" href="/public/css/styles.css">
      <link rel="shortcut icon" type="image/png" href="/public/img/favicon.png">
      <script src="/public/js/jquery-1.9.1.min.js" type="text/javascript" charset="utf-8"></script>
    </head>
    <body>
```

接着，我们在`header.html`中`<body>`标签后加入下面的代码：  

```
  <div class="container main-nav">
    <div class="top-bar">
    <div class="top-bar-inner">
      <ul class="ul-nav ">
          <li class="{{.home}}">
            <a href="/" >Home</a>
            <span class="navbar-news " title="There is an update in the last 1 hours">1</span>
          </li>
          <li class="{{.write}}">
            <a href="/write" title="Put up your blogs">Submission</a>
          </li>
          <li class="{{.mess}}">
            <a href="/message" title="Message Boards">Message</a>
          </li>
          <li class="{{.history}}">
            <a href="/history" title="History blogs">File</a>
          </li>
          <li class="{{.about}}">
            <a href="/about" title="About Me">About Me</a>
          </li>
          <li class="{{.ema}}">
            <a href="/email" title="The emails of the blog's author">Email</a>
          </li>
          <li>
            <a href="#" title="">RSS</a>
          </li>
      </ul>
    </div>
</div>
```

为了闭合div标签，我们需要在 `/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/footer.html`文件中`</body>`之前中加入`</div>`。  

现在让我们看下目前的效果:  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1415timestamp1445331952463.png/wm)  


可以看到菜单栏样子已经出来啦。  

接下来我们接着事先博客内容，编辑文件`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/Index.html`，输入以下内容：  


```
{{set . "title" "Home - GBlog" }}
{{set . "home" "active" }}
{{template "header.html" .}}
<div class="content">
    <div class="cell">
      <div class="cell-subject">
          <div>
            <a href="#" class="cell-subject-title" ><strang>Test size=45</strang></a>
            <a href="#" class="comment-num" title="Comments">10</a>
          </div>
          <div class="subject-infor">
            <span class="label label-success">Author</span>   <span>jov123@163.com</span>  
            <span class="label label-default">Date</span>  2014-04-15 12:25  
            <span class="label label-info">Read</span>  0
          </div>
      </div>
      <div class="cell-text">
          <pre><code>How to say it, a little bit confused, whether it is ID</code></pre>
      </div>
    </div>
</div>
{{template "footer.html" .}}
```

再次刷新下页面，效果如下:  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1415timestamp1445332259525.png/wm)  


目前为止，首页就设计完成啦，下面我们接着事先博客的发布流程。


### 投稿的实现  

新建文件`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/WBlog.html`，输入以下内容：  

```
{{set . "title" "Submission - GBlog"}}
{{set . "write" "active" }}
{{template "header.html" .}}
<div class="content">
    <div class="write-nav">
      <form action="/putblog" method="post" >
          <div class="form-group" >
            <label style="font-weight: bold;">Title</label>
            {{with $field := field "blog.Title" .}}
            <input type="text" id="{{$field.Id}}" name="{{$field.Name}}"  class="form-control" style="width:98%;min-height:28px;" required  value="{{if $field.Flash}}{{$field.Flash}}{{else}}{{$field.Value}}{{end}}">
              <span class="help-inline erro">{{$field.Error}}</span>
            {{end}}
          </div>
          <div class="form-group" >
            <label style="font-weight: bold;">Author</label>
            {{with $field := field "blog.Email" .}}
            <input type="email" id="{{$field.Id}}" name="{{$field.Name}}" class="form-control" style="width:98%;min-height:28px;" placeholder="Enter your email" required value="{{if $field.Flash}}{{$field.Flash}}{{else}}{{$field.Value}}{{end}}">
              <span class="help-inline erro">{{$field.Error}}</span>
            {{end}}
          </div>
          <div class="form-group" >
            <label style="font-weight: bold;">Subject</label>
            {{with $field := field "blog.Subject" .}}
            <textarea class="form-control" id="{{$field.Id}}" name="{{$field.Name}}"  style="width:98%;line-height: 22px;height: 350px;resize: vertical;" required >{{if $field.Flash}}{{$field.Flash}}{{else}}{{$field.Value}}{{end}}</textarea>
              <span class="help-inline erro">{{$field.Error}}</span>
            {{end}}
          </div>
         <button type="submit" class="btn btn-success">Submit</button>
      </form>
    </div>
</div>
{{template "footer.html" .}}
```

这个模板会在`WBlog` action中被用到，我们在`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/App.go`中，加入一个`WBlog`方法：  

```
func (c App) WBlog() revel.Result {
  return c.Render()
}
```

在这个方法中，它绑定了到了 `App`对象上，那怎么样通过url访问到这个方法呢，那就需要配置路由了，在`/home/shiyanlou/revel_mgo_blog/src/GBlog/conf/routes`加入以下路由:  

```
GET     /write                                  App.WBlog
```

其实我们不用添加以上路由也可以通过`http://localhost:9000/App/WBlog`的方式直接访问该方法，这是因为在`routes`文件中配置了以下路由的原因:  

```
*       /:controller/:action                    :controller.:action
```

目前为止`WBlog`的方法就添加完成了，让我们看下效果，点击菜单栏的`投稿`，显示如下：  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1415timestamp1445332837949.png/wm) 。  


### 数据库设计  

在上一节中，我们实现了投稿功能的相应方法，这一小节中我们将实现数据库用于存储博客内容。  

首先，我们创建`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/models`目录， 在`models`目录中，创建`dao.go`文件，内容如下：  

```go
package models

import (
        "gopkg.in/mgo.v2"
)

const (
        DbName            = "ominds"
        BlogCollection    = "blogs"
        CommentCollection = "gb_comments"
        MessageCollection = "gb_messages"
        HistoryCollection = "gb_historys"
        EmailCollection   = "gb_emails"
        BaseYear          = 2014
)

type Dao struct {
        session *mgo.Session
}

func NewDao() (*Dao, error) {
        // mongodb数据库连接
        session, err := mgo.Dial("localhost")
        if err != nil {
                return nil, err
        }
        return &Dao{session}, nil
}
func (d *Dao) Close() {
        d.session.Close()
}

```

以上代码中，我们设置了Mongodb数据库名称，以及各种表名，同时我们设置了博客链接到本地Mongodb数据库。  

目前为止，数据库连接就建立好啦。接下来，让我们在`models`目录下创建文件`blog.go`，输入以下内容：  

```go
  package models
  import (
    "github.com/revel/revel"
    "gopkg.in/mgo.v2/bson"
    "time"
  )
  type Blog struct {
    // Mongodb bson id，类似于主键
    Id bson.ObjectId 
    Email string
    CDate time.Time
    Title string
    Subject string
    // 评论数
    CommentCnt int
    // 阅读数
    ReadCnt int
    // 用于归档
    Year int 
  }

  func (blog *Blog) Validate(v *revel.Validation) {
    v.Check(blog.Title,
      revel.Required{},
      revel.MinSize{1},
      revel.MaxSize{200},
    )
    v.Check(blog.Email,
      revel.Required{},
      revel.MaxSize{50},
    )
    v.Email(blog.Email)
    v.Check(blog.Subject,
      revel.Required{},
      revel.MinSize{1},
    )
  }

  func (dao *Dao) CreateBlog(blog *Blog) error {
    blogCollection := dao.session.DB(DbName).C(BlogCollection)
    //set the time
    blog.Id = bson.NewObjectId()
    blog.CDate = time.Now();
    blog.Year = blog.CDate.Year();
    _, err := blogCollection.Upsert(bson.M{"_id": blog.Id}, blog)
    if err != nil {
      revel.WARN.Printf("Unable to save blog: %v error %v", blog, err)
    }
    return err
  }

  func (dao *Dao) FindBlogs() []Blog{
    blogCollection := dao.session.DB(DbName).C(BlogCollection)
    blogs := []Blog{}
    query := blogCollection.Find(bson.M{}).Sort("-cdate").Limit(50)
    query.All(&blogs)
    return blogs
  }
```

其中 我们首先定义了 `Blog`结构体用于存储博客文章，其映射到Mongodb中。接着我们定义了`Validate`方法用于验证`Blog`的字段，它用到了Revel自带的校验方法。`CreateBlog`方法用于创建博客文章。  

接着我们在`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers`目录中，创建文件`wblog.go`，内容如下：  

```go
package controllers

import (
        "github.com/revel/revel"
        "GBlog/app/models"
        "strings"
)

type WBlog struct {
        App
}

func (c WBlog) Putup(blog *models.Blog) revel.Result {
        blog.Title = strings.TrimSpace(blog.Title)
        blog.Email = strings.TrimSpace(blog.Email)
        blog.Subject = strings.TrimSpace(blog.Subject)
        blog.Validate(c.Validation)
        if c.Validation.HasErrors() {
                c.Validation.Keep()
                c.FlashParams()
                return c.Redirect(App.WBlog)
        }
        dao, err := models.NewDao()
        if err != nil {
                c.Response.Status = 500
                return c.RenderError(err)
        }
        defer dao.Close()
        err = dao.CreateBlog(blog)
        if err != nil {
                c.Response.Status = 500
                return c.RenderError(err)
        }
        return c.Redirect(App.Index)
}
```

现在让我们加入路由吧，在`/home/shiyanlou/revel_mgo_blog/src/GBlog/conf/routes`文件中加入以下内容：  

```
POST    /putblog                                WBlog.Putup
```

为了让我们POST的博客文章能成功显示，我们还需要修改`Index` Action，我们将`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go`修改为如下内容：  

```
import (
  "github.com/revel/revel"
  "GBlog/app/models"
)
type App struct {
  *revel.Controller
}

func (c App) Index() revel.Result {
  dao, err := models.NewDao()
  if err != nil {
    c.Response.Status = 500
    return c.RenderError(err)
  }
  defer dao.Close()
  // 读取所有的博客文章
  blogs := dao.FindBlogs()
  return c.Render(blogs)
}
func (c App) WBlog() revel.Result {
  return c.Render()
}
```

以上代码中，我们从Mongodb中读取所有的博客文章，并在模板中渲染。接着，我们需要修改`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/Index.html`显示文章，其内容调整为如下：  

```
{{set . "title" "Home - GBlog" }}
{{set . "home" "active" }}
{{template "header.html" .}}
<div class="content">
    {{if .blogs}}
    {{range $blog := .blogs}}
     <div class="cell">
      <div class="cell-subject">
          <div>
            <a href="/bloginfor/{{$blog.Id.Hex}}/{{$blog.ReadCnt}}" class="cell-subject-title" title="{{$blog.Title}}"><strang>{{$blog.GetShortTitle }}</strang></a>
            <a href="#" class="comment-num" title="Comments">{{$blog.CommentCnt}}</a>
          </div>
          <div class="subject-infor">
            <span class="label label-success">Author</span>   <span>{{$blog.Email}}
            </span>  
            <span class="label label-default">Date</span>  {{$blog.CDate.Format "2006-01-02 15:04" }}  
            <span class="label label-info">Read</span>  {{$blog.ReadCnt}}
          </div>
      </div>
      <div class="cell-text">
          <pre><code>{{$blog.GetShortContent }}</code></pre>
      </div>
    </div>
    {{end}}
    {{end}}
</div>
{{template "footer.html" .}}
```

以上代码中，我们循环遍历`blogs`对象（是从`Index`中传递过来的博客文章列表）显示文章内容。其中我们用到了两个新方法`GetShortTitle`和`GetShortContent`，在`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/models/blog.go`中加入这两个方法：  

```
func (blog *Blog) GetShortTitle() string{
  if len(blog.Title)>35 {
    return blog.Title[:35]
  }
  return blog.Title
}

func (blog *Blog) GetShortContent() string{
  if len(blog.Subject)>200 {
    return blog.Subject[:200]
  }
  return blog.Subject
}
```

到这里，博客发布文章的功能就全部完成啦。下面让我们进行测试。在应用中我们用到了Mongodb数据库，所以我们首先要启动Mongodb，输入以下命令启动Mongodb：

```
$ sudo mongod --fork -f /etc/mongodb.conf
```

现在我们重新刷新页面点击投稿试试看：  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1415timestamp1445394190222.png/wm)  


接着点击`Submit`按钮，现在首页就有我们的文章了：  

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1415timestamp1445394268053.png/wm)  


同时查看Mongodb，可以看到数据都存储进去了：  

```
$ mongo                                                                                                                         
MongoDB shell version: 2.4.9
connecting to: test
> show dbs;
local   0.078125GB
ominds  0.203125GB
> use ominds;
switched to db ominds
> show collections
blogs
system.indexes
> db.blogs.find()
{ "_id" : ObjectId("5626f6a441675672310965a2"), "id" : ObjectId("5626f6a44b3b34108f6428fb"), "email" : "support@shiyanlou.com", "cdate" : ISODate("2015-10-21T02:21:24.541Z"), "title" : "The first article", "subject" : "Our blog has a release function.", "commentcnt" : 0, "readcnt" : 0, "year" : 2015 }
```

到这里，博客的发布功能就全部完成了  

## 评论和留言功能实现  

这一节实验中，主要讲解怎么样实现博客的评论和留言功能。  

本实验属于[实验楼](https://www.shiyanlou.com)课程《基于Reve和mgo的博客》,该课程基于[joveth](http://blog.csdn.net/joveth)的教程[Go web开发之revel+mgo](http://blog.csdn.net/jov123/article/category/2216585)改编而来。具体来说，在[joveth](http://blog.csdn.net/joveth)教程的基础上，实验楼修改了教程以适配最新版本的 `Revel`。  

为了能正常开始实验，我们需要设置 `GOPATH` 环境变量以及启动mongodb，每次开始实验前，请执行以下命令：  

```
$ cd /home/shiyanlou/revel_mgo_blog
$ export GOPATH=`pwd`
$ export PATH=$PATH:$GOPATH/bin
$ sudo mongod --fork -f /etc/mongodb.conf
```

## 全局样式

为了方便使用，这里将整个项目的css贴出，将以下内容写入到`/home/shiyanlou/revel_mgo_blog/src/GBlog/public/css/styles.css` 中：

```
    body{
      margin: 0 auto;
      padding: 0;
      font: 14px "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif;
      line-height: 20px;
      letter-spacing: 0.02em;
      color: #666;
      background-attachment:fixed;
    }
    a{
      color: #1abc9c;
      text-decoration: none;
      -webkit-transition: 0.25s;
      -moz-transition: 0.25s;
      -o-transition: 0.25s;
      transition: 0.25s;
      -webkit-backface-visibility: hidden;
    }
    .main-nav{
      margin: 0 auto;
      width: 692px;
      padding:0;
    }
    .top-bar{
      width:100%;
      background: #34495e;
      border-bottom-right-radius: 6px;
      border-bottom-left-radius: 6px;
      box-shadow:  0 2px rgba(0,0,0,0.075),0 0 6px #7aba7b;
      -webkit-box-shadow:0 2px  rgba(0,0,0,0.075),0 0 6px #7aba7b;
      -moz-box-shadow:0 2px  rgba(0,0,0,0.075),0 0 6px #7aba7b;
      margin-bottom:28px;
    }
    .top-bar-inner{
      min-height: 48px;
      padding:0 4px;
    }
    .ul-nav{
      position: relative;
      left: 0;
      display: block;
      float: left;
      margin: 0 10px 0 0;
      list-style: none;
      font-size: 18px;
      padding:0;
    }
    .ul-nav>li {
      position: relative;
      float: left;
      line-height: 20px;
    }
    .ul-nav>li>a{
      padding: 14px 24px 17px;
      text-decoration: none;
      display: block;
      color: white;
      text-shadow: 0 -1px 0 rgba(0,0,0,0.25);
    }
    .ul-nav>li>a:hover,.ul-nav>li>a:focus{
      color: #1abc9c;
    }
    .navbar-news {
      background-color: #e74c3c;
      border-radius: 30px;
      color: white;
      display: block;
      font-size: 12px;
      font-weight: 500;
      line-height: 18px;
      min-width: 8px;
      padding: 0 5px;
      position: absolute;
      right: -7px;
      text-align: center;
      text-shadow: none;
      top: 8px;
      z-index: 10;
    }
    .ul-nav .active > a, .ul-nav .active > a:hover, .ul-nav .active > a:focus {
      background-color: transparent;
      color: #1abc9c;
      -webkit-box-shadow: none;
      -moz-box-shadow: none;
      box-shadow: none; 
    }
    .cell{
      background-color:#1bc6a5;
      color: #cff3ec;
      font-size: 15px;
      border-radius: 4px;
      -webkit-border-radius: 4px;
      -moz-border-radius: 4px;
      -o-border-radius: 4px;
      -khtml-border-radius: 4px;
      padding: 18px 20px 0px 20px; 
      margin-bottom: 30px;
      box-shadow: 0 1px 1px rgba(0, 0, 0, 0.2);

    }
    .cell-subject{
      margin: 0;
    }
    .cell-subject-title{
      color: #34495e;
      font-size: 24px;
      font-weight: 700;
      text-decoration: none;
    }
    a.cell-subject-title:hover{
      text-decoration: underline;
    }
    .subject-infor{
      color:#34495e;
      line-height: 19px;
      padding: 2px 0;
      font-size: 13px;
      margin:2px 0;
    }
    .cell-text{
      padding: 4px 0;
      word-break: break-all;
    }
    .comment-num{
      float:right;
      border: 5px solid #d7dce0;
      border-radius: 50px;
      font-size: 14px;
      line-height: 16px;
      padding: 0 4px;
      -webkit-transition: background 0.2s ease-out, border-color 0s ease-out, color 0.2s ease-out;
      -moz-transition: background 0.2s ease-out, border-color 0s ease-out, color 0.2s ease-out;
      -o-transition: background 0.2s ease-out, border-color 0s ease-out, color 0.2s ease-out;
      transition: background 0.2s ease-out, border-color 0s ease-out, color 0.2s ease-out;
      -webkit-backface-visibility: hidden;
      background-color: white;
      border-color: white;
      border-width: 2px;
      color: #BBB6B6;
    }
    .write-nav{
      padding: 10px;
      background-color: #f9f9f9;
      border: 1px solid #e4e4e4;
      border-radius: 5px;
    }
    .footer{
      margin:20px 0 10px 0;
      text-align: center;
      min-height: 20px;
    }
    .comment-nav,.history-nav{
      background-color: white;
      border: 1px solid #DDD;
      border-radius: 4px;
    }
    .comment-title,.history-title{
      padding: 10px 20px;
      border-bottom: 1px solid #e2e2e2;
      font-size: 15px;
      font-weight: bold;
    }
    .comment-cell{
      padding: 10px 10px;
      border-top: 2px solid #fff;
      border-bottom: 1px solid #e2e2e2;
    }
    .comment-inputbox{
      background-color: #E7E7D8;
      border: 1px solid #d6d6c6;
      padding:5px 10px;
      border-radius: 8px;
      height: 224px;

    }
    .comment-input-infor{
      float: left;
      width: 180px;
      height: 200px;
      display: inline;
      overflow: hidden;
      margin:5px 0  0 0;
      list-style: none;
      padding:0;
    }

    .comment-input-text{
      float: left;
      width: 458px;
      height: 150px;
      margin:5px 0  0 10px;
    }
    .ctextarea {
      height: 142px !important;
      resize: vertical;
    }
    .comment-input-text-btn{
      float:right;
      margin-top:20px;
    }
    .func-color{
      color: #F70246;
    }
    .func-name{
      color: #24C54B;
    }
    .func-str{
      color:#C29916;
    }
    .func-type{
      color:rgb(0, 173, 255);
    }
    .pln-w{
      color:white;
    }
    .pln-b{
      color:white;
    }
    .history-cell{
      padding: 10px 20px;
      border-top: 2px solid #fff;
      border-bottom: 1px solid #e2e2e2;
    }
    li time{
      margin-right: 8px;
      font-size: 13px;
    }
    .history-auth{
      padding-left: 8px;
      color: #cfcfcf;
      font-size: 13px;
      font-family: 'Microsoft Yahei';
    }
    .email-other{
      background-color: #f9f9f9;
      border: 1px solid #e4e4e4;
      border-radius: 5px;
      padding: 5px 20px;
      margin-bottom: 10px;
    }
    .email-title{
      margin: 0 0 22px;
      padding-top: 21px;
      color: white;
      font-family: 'Microsoft Yahei';
    }
    .email-nav ul{
      list-style-type: none;
      margin: 0 0 26px;
      padding:0;
    }
    .email-nav ul li:first-child {
      border-top: none;
      padding-top: 1px;
    }
    .email-nav ul li {
      border-top: 1px solid #1bc6a5;
      line-height: 19px;
      padding: 6px 0;
    }
    .email-tag{
      border-radius: 4px;
      background: #1abc9c;
      color: white;
      cursor: pointer;
      margin-right: 5px;
      margin-bottom: 5px;
      overflow: hidden;
      padding: 6px 13px 6px 19px;
      position: relative;
      vertical-align: middle;
      display: inline-block;
      zoom: 1;
      -webkit-transition: 0.14s linear;
      -moz-transition: 0.14s linear;
      -o-transition: 0.14s linear;
      transition: 0.14s linear;
      -webkit-backface-visibility: hidden;
    }
    .email-tag span{
      color: white;
      cursor: pointer;
      zoom: 1;
    }
    .email-nav{
      background: white;
      border: 2px solid #1abc9c;
      border-radius: 6px;
      padding: 6px 1px 1px 6px;
      overflow-y: auto;
      text-align: left;
      margin-bottom: 15px;
    }
    .email-tag:hover {
      background-color: #16a085;
      padding-left: 12px;
      padding-right: 20px;
    }
    .erro{
      color:red;
    }
    .infor-content,.comments{
      margin-bottom: 20px;
      padding: 10px 20px;
      background-color: white;
      border: 1px solid #DDD;
      border-radius: 4px;
      -webkit-border-radius: 4px;
      -moz-border-radius: 4px;
      -o-border-radius: 4px;
      -khtml-border-radius: 4px;
    }
    .infor-header{
      padding-bottom: 9px;
      margin: 0 0 20px;
      border-bottom: 1px solid #eee;
    }
    .infor-header h3{
      margin-top:10px;
    }
    .infor-body{
      padding-bottom: 9px;
    }
    hr {
      margin: 9px 0;
    }
    dl.the-comments dd{
      margin-left: 0;
      padding-top: 10px;
      padding-bottom: 10px;
      border-bottom: 1px dashed #CCC;
    }
    .user-comment p{
      margin:5px 0 0 0;
    }

```

## 详情页和评论功能

### 设计博客章详情页面

在 `/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App`目录下创建文件`BlogInfor.html`，输入以下内容：

```
{{set . "title" "Bloginfor - GBlog" }}
{{set . "home" "active" }}
{{template "header.html" .}}
<div class="content">
    <div class="infor-content">
        <div class="infor-header">
          <h3>Title</h3>
          <div class="subject-infor">
            <span class="label label-success">Author</span>   <span>jov123@163.com</span>  
            <span class="label label-default">Date</span>  2014-04-25 15:04  
            <span class="label label-info">Read</span>  1
          </div>
        </div>
        <div class="infor-body">
          this is the subject
        </div>
    </div>
    <div class="comments">
        <span>Reply</span>
        <hr>
        <dl class="the-comments">
          <dd >
            <span class="label label-default pull-right">#1</span>
            <div class="user-info">
              <a href="#"><strong>omind@163.com</strong></a> •
              2014-04-25 16:04
            </div>
            <div class="user-comment">
              <p>nice!</p>
            </div>
          </dd>
        </dl>
    </div>
    <div class="comments">
        <div class="comment-form">
          <form action="/docomment" method="post">
            <input type="hidden" name="id" value="{{.blog.Id.Hex}}">
            <input type="hidden" name="rcnt" value="{{.rcnt}}">
            <div class="form-group">
              <label >Email</label>
              {{with $field := field "comment.Email" .}}
              <input type="email" class="form-control" id="{{$field.Id}}" name="{{$field.Name}}"  placeholder="Your email" required value="{{if $field.Flash}}{{$field.Flash}}{{else}}{{$field.Value}}{{end}}">
              <span class="help-inline erro">{{$field.Error}}</span>
              {{end}}
            </div>
            <div class="form-group">
              <label >Comment</label>
              {{with $field := field "comment.Content" .}}
              <textarea class="form-control" id="{{$field.Id}}" name="{{$field.Name}}" rows="6" placeholder="Enter the comment" required >{{if $field.Flash}}{{$field.Flash}}{{else}}{{$field.Value}}{{end}}</textarea>
              {{end}}
            </div>
            <div class="form-group">
              <button type="submit" class="btn btn-success">Submit</button>
            </div>
          </form>
        </div>
    </div>
</div>
{{template "footer.html" .}}
```

接着添加相应的的controller和routes，编辑文件`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go` 添加以下方法：

```
func (c App) BlogInfor() revel.Result {
  return c.Render()
}
```
在`/home/shiyanlou/revel_mgo_blog/src/GBlog/conf/routes`中加入以下路由：

```
GET     /bloginfor                              App.BlogInfor
```
现在访问http://localhost:9000/bloginfor访问看看，效果如下：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1416timestamp1445398679927.png/wm)

### 实现博客详情页

在 `/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/Index.html`页面中，有以下代码：

```
<a href="/bloginfor/{{$blog.Id.Hex}}/{{$blog.ReadCnt}}" class="cell-subject-title" title="{{$blog.Title}}"><strang>{{$blog.GetShortTitle }}</strang></a>
```

其中，博客文章的链接是通过`/bloginfor/{{$blog.Id.Hex}}/{{$blog.ReadCnt}}`方式生成的，我们传递了`Blog`对象的id和阅读次数， 用于增加文章的阅读数。

接着调整 `/home/shiyanlou/revel_mgo_blog/src/GBlog/conf/routes` 中路由条目`App.BlogInfor`为以下内容，以增加参数配置：

```
 GET     /bloginfor/:id/:rcnt                    App.BlogInfor
```

同时也要调整`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go`中的`BlogInfor`方法，以接收参数，调整为如下：

```
func (c App) BlogInfor(id string,rcnt int) revel.Result {
  dao, err := models.NewDao()
  if err != nil {
    c.Response.Status = 500
    return c.RenderError(err)
  }
  defer dao.Close()
  blog := dao.FindBlogById(id)
  if(blog.ReadCnt==rcnt){
    blog.ReadCnt = rcnt+1
    dao.UpdateBlogById(id,blog)
  }
  return c.Render(blog, rcnt)
}
```

以上方法中，我们首先根据Blog ID在数据库中查找文章，找到后判断参数`rcnt`是否和数据库中几率的阅读数一致，如果一致的话，则增加阅读数一次。同时也看到，我们使用了两个新的方法`FindBlogById`和`UpdateBlogById`，编辑`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/models/blog.go`增加下面两个方法：

```
  func (dao *Dao) FindBlogById(id string) *Blog{
    blogCollection := dao.session.DB(DbName).C(BlogCollection)
    blog := new(Blog)
    query := blogCollection.Find(bson.M{"id": bson.ObjectIdHex(id)})
    query.One(blog)
    return blog
  }
  
  func (dao *Dao) UpdateBlogById(id string,blog *Blog) {
    blogCollection := dao.session.DB(DbName).C(BlogCollection)
    err := blogCollection.Update(bson.M{"id": bson.ObjectIdHex(id)}, blog)
    if err!=nil{
      revel.WARN.Printf("Unable to update blog: %v error %v", blog, err)
    }
  }

```

同时需要再次编辑 `/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/BlogInfor.html`文件，替换成下面的内容：

```
 {{set . "title" "Bloginfor - GBlog" }}
    {{set . "home" "active" }}
    {{template "header.html" .}}
    <div class="content">
        {{if .blog}}
        <div class="infor-content">
            <div class="infor-header">
              <h3>{{.blog.Title}}</h3>
              <div class="subject-infor">
                <span class="label label-success">Author</span>   <span>{{.blog.Email}}</span>  
                <span class="label label-default">Date</span>  {{.blog.CDate.Format "2006-01-02 15:04"}}  
                <span class="label label-info">Read</span>  {{.blog.ReadCnt}}
              </div>
            </div>
            <div class="infor-body">
              {{.blog.Subject}}
            </div>
        </div>
        <div class="comments">
            <span>Reply</span>
            <hr>
            <dl class="the-comments">
              <dd >
                <span class="label label-default pull-right">#1</span>
                <div class="user-info">
                  <a href="#"><strong>omind@163.com</strong></a> •
                  2014-04-25 16:04
                </div>
                <div class="user-comment">
                  <p>nice!</p>
                </div>
              </dd>
            </dl>
        </div>
        <div class="comments">
            <div class="comment-form">
              <form action="/docomment" method="post">
                <input type="hidden" name="id" value="{{.blog.Id.Hex}}">
                <input type="hidden" name="rcnt" value="{{.rcnt}}">
                <div class="form-group">
                  <label >Email</label>
                  {{with $field := field "comment.Email" .}}
                  <input type="email" class="form-control" id="{{$field.Id}}" name="{{$field.Name}}"  placeholder="Your email" required value="{{if $field.Flash}}{{$field.Flash}}{{else}}{{$field.Value}}{{end}}">
                  <span class="help-inline erro">{{$field.Error}}</span>
                  {{end}}
                </div>
                <div class="form-group">
                  <label >Comment</label>
                  {{with $field := field "comment.Content" .}}
                  <textarea class="form-control" id="{{$field.Id}}" name="{{$field.Name}}" rows="6" placeholder="Enter the comment" required >{{if $field.Flash}}{{$field.Flash}}{{else}}{{$field.Value}}{{end}}</textarea>
                  {{end}}
                </div>
                <div class="form-group">
                  <button type="submit" class="btn btn-success">Submit</button>
                </div>
              </form>
            </div>
        </div>
        {{end}}
    </div>
    {{template "footer.html" .}}
```

现在我们可以在首页点击某一个博客文章标题，现在能跳转到相应的文章详情页面了。


#### 实现评论功能

评论肯定也是需要存储到数据库中，这就需要我们实现model。在`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/models`目录中，创建文件`comment.go`，输入以下内容：

```
package models

import (
  "github.com/revel/revel"
  "gopkg.in/mgo.v2/bson"
  "time"
)

type Comment struct{
  BlogId bson.ObjectId 
  Email string
  CDate time.Time
  Content string
}

func (comment *Comment) Validate(v *revel.Validation) {
  v.Check(comment.Email,
    revel.Required{},
    revel.MaxSize{50},
  )
  v.Email(comment.Email)
  v.Check(comment.Content,
    revel.Required{},
    revel.MinSize{1},
    revel.MaxSize{1000},
  )
}

// 插入评论
func (dao *Dao) InsertComment(comment *Comment) error {
  commCollection := dao.session.DB(DbName).C(CommentCollection)
  //set the time
  comment.CDate = time.Now();
  err := commCollection.Insert(comment)
  if err != nil {
    revel.WARN.Printf("Unable to save Comment: %v error %v", comment, err)
  }
  return err
}

// 查找评论
func (dao *Dao) FindCommentsByBlogId(id bson.ObjectId) []Comment{
  commCollection := dao.session.DB(DbName).C(CommentCollection)
  comms := []Comment{}
  query := commCollection.Find(bson.M{"blogid":id}).Sort("CDate")
  query.All(&comms)
  return comms
}
```

以上代码和`blog.go`代码非常相似，定义了`Comment`结构体用于存储评论，以及一系列查找编辑方法。

我们要在`BlogInfor.html`页面里提交评论，肯定需要一个方法来接收这个评论的，下面就让我们来实现它，在`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers`目录中创建文件`wcomment.go`，输入以下内容：

```
 package controllers

  import (
    "github.com/revel/revel"
    "GBlog/app/models"
    "strings"
  )
  type WComment struct {
    App
  }
  func (c WComment) Docomment(id string,rcnt int,comment *models.Comment) revel.Result {
    if len(id)==0{
      return c.Redirect(App.Index)
    }
    dao, err := models.NewDao()
    if err != nil {
      c.Response.Status = 500
      return c.Redirect(App.Index)
    }
    defer dao.Close()
    blog := dao.FindBlogById(id)
    if blog==nil {
      return c.Redirect(App.Index)
    }
    comment.BlogId = blog.Id
    comment.Content = strings.TrimSpace(comment.Content)
    comment.Email = strings.TrimSpace(comment.Email)
    comment.Validate(c.Validation)
    if c.Validation.HasErrors() {
      c.Validation.Keep()
      c.FlashParams()
      c.Flash.Error("Errs:The email and the content should not be null,or the maxsize of email is 50.")
      return c.Redirect("/bloginfor/%s/%d",id,rcnt)
    }
    err = dao.InsertComment(comment)
    if err!=nil {
      c.Response.Status = 500
      return c.RenderError(err)
    }
    blog.CommentCnt++
    dao.UpdateBlogById(id,blog)
    return c.Redirect("/bloginfor/%s/%d",id,rcnt)
  }

```

以上代码中，我们首先根据参数`id`查找`Blog`对象，然后插入相应的评论。同时我们需要添加路由，编辑`/home/shiyanlou/revel_mgo_blog/src/GBlog/conf/routes`加入以下路由：

```
POST    /docomment                                WComment.Docomment
```

现在为止我们可以提交评论啦，但是我们还需要在详情页面中显示评论，为了显示命令，我们需要博客的评论传递到模板中，同时在模板中遍历显示评论。下面让我们来实现。

手下在`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go`里的`BlogInfor`方法里，在`return`关键字之前添加以下代码：

```
  comments := dao.FindCommentsByBlogId(blog.Id);
if len(comments)==0&&blog.CommentCnt!=0{
    blog.CommentCnt=0;
    dao.UpdateBlogById(id,blog)
}else if len(comments)!=blog.CommentCnt{
    blog.CommentCnt=len(comments);
    dao.UpdateBlogById(id,blog)
}

```
同时将`return`语句修改为：


```
return c.Render(blog,rcnt,comments)

```

同时在模板中遍历评论，编辑`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/BlogInfor.html`，将其中的块：

```
 <div class="comments">
    <span>Reply</span>
    <hr>
    <dl class="the-comments">
      <dd >
        <span class="label label-default pull-right">#1</span>
        <div class="user-info">
          <a href="#"><strong>omind@163.com</strong></a> •
          2014-04-25 16:04
        </div>
        <div class="user-comment">
          <p>nice!</p>
        </div>
      </dd>
    </dl>
</div>
```

修改为：

```
{{if .comments}}
<div class="comments">
    <span>Reply</span>
    <hr>
    <dl class="the-comments">
       {{range $index,$comment := .comments}}
      <dd >
        <span class="label label-default pull-right">#{{pls $index 1}}</span>
        <div class="user-info">
          <a href="#"><strong>{{$comment.Email}}</strong></a> •
          {{$comment.CDate.Format "2006-01-02 15:04" }}
        </div>
        <div class="user-comment">
          <p>{{$comment.Content}}</p>
        </div>
      </dd>
      {{end}}
    </dl>
</div>
{{end}}
```

在上面的代码中，我们使用到了`{{pls $index }}`，其中`pls`，我们定义的一个模板函数, 用于显示评论的楼层。下面在 `/home/shiyanlou/revel_mgo_blog/src/GBlog/app/init.go`中的 `init`方法里添加如下代码：


```
revel.TemplateFuncs["pls"] = func(a, b int) int { return a + b }
```

再次刷新页面，你就能看到评论功能工作正常啦，效果如下：


![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1416timestamp1445411985963.png/wm)



## 留言功能

### 设计留言板页面

在 `/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App`目录中创建文件`Message.html`，输入以下内容：

```
  {{set . "title" "Message - GBlog"}}
  {{set . "mess" "active" }}
  {{template "header.html" .}}
  <div class="content">
      <div class="comment-nav">
         <div class="comment-title">
           Message Board
           <span style="float:right" title="Total messages">[100]</span>
         </div>
         <div class="comment-cell">
           <div class="comment-inputbox">
              <form action="/putmessage" method="post" >
                    <ul class="comment-input-infor">
                      {{with $field := field "message.Email" .}}
                      <li>
                        EMAIL
                        <input type="email"  id="{{$field.Id}}" name="{{$field.Name}}" class="form-control " required value="{{if $field.Error}}{{$field.Error}}{{else}}{{$field.Value}}{{end}}"/>
                      </li>
                       {{end}}
                      <li>
                        QQ
                        {{with $field := field "message.QQ" .}}
                        <input type="text"  id="{{$field.Id}}" name="{{$field.Name}}" class="form-control" value="{{if $field.Error}}{{$field.Error}}{{else}}{{$field.Value}}{{end}}" />
                        {{end}}
                      </li>
                      <li>
                        Personal home page
                        {{with $field := field "message.Url" .}}
                        <input type="text"  id="{{$field.Id}}" name="{{$field.Name}}" class="form-control" placeholder="Don't with http://  " value="{{if $field.Error}}{{$field.Error}}{{else}}{{$field.Value}}{{end}}"/>
                        {{end}}
                      </li>
                    </ul>
                    <div class="comment-input-text">
                      MESSAGE
                      <div>
                        {{with $field := field "message.Content" .}}
                        <textarea class="form-control ctextarea" id="{{$field.Id}}" name="{{$field.Name}}" required>{{if $field.Error}}{{$field.Error}}{{else}}{{$field.Value}}{{end}}</textarea>
                        {{end}}
                      </div>
                    </div>
                    <button type="submit" class="btn btn-success comment-input-text-btn">SUBMIT</button>
              </form>
           </div>
         </div>
         <div class="comment-cell">
           <pre ><code><span class="func-color">func</span><span class="func-name"> UserMessage</span><span class="pln">() </span><span class="pun">{</span><span class="pln">
      </span><span class="func-color">var</span><span class="pun"> </span><span class="pln">email </span><span class="func-type">string </span><span class="pun">=</span><span class="pln"> </span><span class="func-str">"jov123@163.com"</span><span class="pun">;</span><span class="pln-w">
      </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln">url </span><span class="pln">=</span><span class="pln-w"> </span><a href="http://jov.herokuapp.com" class="func-str" target="_blank">http://jov.herokuapp.com</a><span class="pln-w">;</span><span class="pln">
      </span><span class="lit">Date </span><span class="func-color">:=</span><span class="pln"> </span><span class="func-str">"2014-04-15 12:50"</span><span class="pun">;</span><span class="pln">
      </span><span class="func-color">var </span><span class="lit">Message</span><span class="pln"> </span><span class="pun">=</span><span class="pln"> </span><span class="func-str">"nice!"</span><span class="pln">;
  </span><span class="pun">}</span><span class="pln">
  </code></pre>
         </div>
          <div class="comment-cell">
           <pre style="background:#2d2d2d;"><code><span class="func-color">func</span><span class="func-name"> UserMessage</span><span class="pln-w">() </span><span class="pln-w">{</span><span class="pln-w">
      </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln-w">email </span><span class="func-type">string </span><span class="pln-w">=</span><span class="pln-w"> </span><span class="func-str">"jov123@163.com"</span><span class="pln-w">;</span><span class="pln-w">
      </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln-w">url </span><span class="pln-w">=</span><span class="pln-w"> </span><a href="http://jov.herokuapp.com" class="func-str" target="_blank">http://jov.herokuapp.com</a><span class="pln-w">;</span><span class="pln-w">
      </span><span class="pln-w">Date </span><span class="func-color">:=</span><span class="pln-w"> </span><span class="func-str">"2014-04-15 12:50"</span><span class="pln-w">;</span><span class="pln-w">
      </span><span class="func-color">var </span><span class="pln-w">Message</span><span class="pln-w"> </span><span class="pln-w">=</span><span class="pln-w"> </span><span class="func-str">"nice!"</span><span class="pln-w">;
  </span><span class="pln-w">}</span><span class="pln-w">
  </code></pre>
         </div>
    </div>
  </div>
  {{template "footer.html" .}}
```

接着添加相应的的controller和路由，编辑 `/home/shiyanlou/revel_mgo_blog/src/GBlog/conf/routes`文件，添加以下路由：

```
GET     /message                                App.Message
```

在`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go`里添加方法：

```
func (c App) Message() revel.Result {
  return c.Render()
}
```

然后点击菜单栏的留言就可以看到效果了。现在只是一个展示页面，下面让我们真正实现它。


### 实现留言板功能

类似于评论功能的实现，我们需要依次实现model，controller等。首先在 ``下新建文件`message.go`，输入以下内容：

```
package models
  import (
    "github.com/revel/revel"
    "gopkg.in/mgo.v2/bson"
    "time"
  )
  type Message struct{
    Email string
    QQ string
    Url string
    CDate time.Time
    Content string
  }
  func (message *Message) Validate(v *revel.Validation) {
    v.Check(message.Email,
      revel.Required{},
      revel.MaxSize{50},
    )
    v.Email(message.Email)
    v.Check(message.QQ,
      revel.MaxSize{20},
    )
    v.Check(message.Url,
      revel.MaxSize{200},
    )
    v.Check(message.Content,
      revel.Required{},
      revel.MinSize{1},
      revel.MaxSize{1000},
    )
  }
  func (dao *Dao) InsertMessage(message *Message) error {
    messCollection := dao.session.DB(DbName).C(MessageCollection)
    //set the time
    message.CDate = time.Now();
    err := messCollection.Insert(message)
    if err != nil {
      revel.WARN.Printf("Unable to save Message: %v error %v", message, err)
    }
    return err
  }
  func (dao *Dao) FindAllMessages() []Message{
    messCollection := dao.session.DB(DbName).C(MessageCollection)
    mess := []Message{}
    query := messCollection.Find(bson.M{}).Sort("-cdate")
    query.All(&mess)
    return mess
  }

```

接着添加相应的controller，在``目录下新建`wmessage.go`，输入以下内容：

```
package controllers

  import (
    "github.com/revel/revel"
    "GBlog/app/models"
    "strings"
    "fmt"
  )
  type WMessage struct {
    App
  }
  func (c WMessage) Putup(message *models.Message) revel.Result {
    message.Email = strings.TrimSpace(message.Email);
    message.Url = strings.TrimSpace(message.Url);
    message.Content = strings.TrimSpace(message.Content);
    message.QQ = strings.TrimSpace(message.QQ);
    message.Validate(c.Validation)
    fmt.Println(c.Validation)
    if c.Validation.HasErrors() {
      c.Validation.Keep()
      c.FlashParams()
      c.Flash.Error("Errs:The email and the content should not be null,or the maxsize of email is 50.")
      return c.Redirect(App.Message)
    }
    dao, err := models.NewDao()
    if err != nil {
      c.Response.Status = 500
      return c.RenderError(err)
    }
    defer dao.Close()
    err = dao.InsertMessage(message)
    if(err!=nil){
      c.Response.Status = 500
      return c.RenderError(err)
    }
    return c.Redirect(App.Message)
  }
```

然后再`/home/shiyanlou/revel_mgo_blog/src/GBlog/conf/routes`中添加以下路由：

```
POST   /putmessage                           WMessage.Putup
```

经过以上步骤，我们发布留言的功能就实现了，但是怎么显示留言呢？这就需要修改之前设计的`Mesage()`方法和`Message.html`页面了。

首先修改`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go`中的`Mesasge()`方法为以下内容：

```
  func (c App) Message() revel.Result {
    dao, err := models.NewDao()
    if err != nil {
      c.Response.Status = 500
      return c.RenderError(err)
    }
    defer dao.Close()
    //dao := models.NewDao(c.MongoSession)
    messages := dao.FindAllMessages()
    return c.Render(messages)
  }
```
接着调整页面`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/Message.html`为以下内容：

```
    {{set . "title" "Message - GBlog"}}
    {{set . "mess" "active" }}
    {{template "header.html" .}}
    <div class="content">
        <div class="comment-nav">
           <div class="comment-title">
             Message Board
             <span style="float:right" title="Total messages">[{{len .messages}}]</span>
           </div>
           <div class="comment-cell">
             <div class="comment-inputbox">
                <form action="/putmessage" method="post" >
                      <ul class="comment-input-infor">
                        {{with $field := field "message.Email" .}}
                        <li>
                          EMAIL
                          <input type="email"  id="{{$field.Id}}" name="{{$field.Name}}" class="form-control " required value="{{if $field.Error}}{{$field.Error}}{{else}}{{$field.Value}}{{end}}"/>
                        </li>
                         {{end}}
                        <li>
                          QQ
                          {{with $field := field "message.QQ" .}}
                          <input type="text"  id="{{$field.Id}}" name="{{$field.Name}}" class="form-control" value="{{if $field.Error}}{{$field.Error}}{{else}}{{$field.Value}}{{end}}" />
                          {{end}}
                        </li>
                        <li>
                          Personal home page
                          {{with $field := field "message.Url" .}}
                          <input type="text"  id="{{$field.Id}}" name="{{$field.Name}}" class="form-control" placeholder="Don't with http://  " value="{{if $field.Error}}{{$field.Error}}{{else}}{{$field.Value}}{{end}}"/>
                          {{end}}
                        </li>
                      </ul>
                      <div class="comment-input-text">
                        MESSAGE
                        <div>
                          {{with $field := field "message.Content" .}}
                          <textarea class="form-control ctextarea" id="{{$field.Id}}" name="{{$field.Name}}" required>{{if $field.Error}}{{$field.Error}}{{else}}{{$field.Value}}{{end}}</textarea>
                          {{end}}
                        </div>
                      </div>
                      <button type="submit" class="btn btn-success comment-input-text-btn">SUBMIT</button>
                </form>
             </div>
           </div>
           {{if .messages}}
           {{range $index,$message:=.messages}}
           {{if mo $index 2}}
           <div class="comment-cell">
             <pre ><code><span class="func-color">func</span><span class="func-name"> UserMessage</span><span class="pln">() </span><span class="pln">{</span><span class="pln-w">
        </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln">email </span><span class="func-type">string </span><span class="pln">=</span><span class="pln"> </span><span class="func-str">"{{$message.Email}}"</span><span class="pln">;</span>{{if $message.QQ}}<span class="pln-w">
        </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln">QQ </span><span class="pln">=</span><span class="pln-w"> </span><span class="func-str" >{{$message.QQ}}</span><span class="pln">;</span>{{end}}{{if $message.Url}}<span class="pln-w">
        </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln">url </span><span class="pln">=</span><span class="pln-w"> </span><a href="http://{{$message.Url}}" class="func-str" target="_blank">http://{{$message.Url}}</a><span class="pln">;</span>{{end}}<span class="pln-w">
        </span><span class="pln">Date </span><span class="func-color">:=</span><span class="pln-w"> </span><span class="func-str">"{{$message.CDate.Format "2006-01-02 15:04"}}"</span><span class="pln">;</span><span class="pln-w">
        </span><span class="func-color">var </span><span class="pln">Message</span><span class="pln-w"> </span><span class="pln">=</span><span class="pln-w"> </span><span class="func-str">"{{$message.Content}}"</span><span class="pln">;
    </span><span class="pln">}</span><span class="pln-w">
    </code></pre>
           </div>
           {{else}}
           <div class="comment-cell">
             <pre style="background:#2d2d2d;"><code><span class="func-color">func</span><span class="func-name"> UserMessage</span><span class="pln-w">() </span><span class="pln-w">{</span><span class="pln-w">
        </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln-w">email </span><span class="func-type">string </span><span class="pln-w">=</span><span class="pln-w"> </span><span class="func-str">"{{$message.Email}}"</span><span class="pln-w">;</span>{{if $message.QQ}}<span class="pln-w">
        </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln-w">QQ </span><span class="pln-w">=</span><span class="pln-w"> </span><span class="func-str" >{{$message.QQ}}</span><span class="pln-w">;</span>{{end}}{{if $message.Url}}<span class="pln-w">
        </span><span class="func-color">var</span><span class="pln-w"> </span><span class="pln-w">url </span><span class="pln-w">=</span><span class="pln-w"> </span><a href="http://{{$message.Url}}" class="func-str" target="_blank">http://{{$message.Url}}</a><span class="pln-w">;</span>{{end}}<span class="pln-w">
        </span><span class="pln-w">Date </span><span class="func-color">:=</span><span class="pln-w"> </span><span class="func-str">"{{$message.CDate.Format "2006-01-02 15:04"}}"</span><span class="pln-w">;</span><span class="pln-w">
        </span><span class="func-color">var </span><span class="pln-w">Message</span><span class="pln-w"> </span><span class="pln-w">=</span><span class="pln-w"> </span><span class="func-str">"{{$message.Content}}"</span><span class="pln-w">;
    </span><span class="pln-w">}</span><span class="pln-w">
    </code></pre>
           </div>
           {{end}}
           {{end}}
           {{end}}
      </div>
    </div>
    {{template "footer.html" .}}
```

在`Message.html`中，我们用到了`{{if mo $index 2}}`这段代码，其中`mo`我们自定义的模板函数，需要在 `/home/shiyanlou/revel_mgo_blog/src/GBlog/app/init.go`中`init`方法中添加以下代码后才能使用：

```
revel.TemplateFuncs["mo"] = func(a, b int) bool { return a%b==0 }

```

到这里整个留言板功能就完成啦，效果如下：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1416timestamp1445414212379.png/wm)  

## 归档功能和提醒功能  

这一节实验中，主要讲解怎么样实现博客的评论和留言功能。

本实验属于[实验楼](https://www.shiyanlou.com)课程《基于Reve和mgo的博客》,该课程基于[joveth](http://blog.csdn.net/joveth)的教程[Go web开发之revel+mgo](http://blog.csdn.net/jov123/article/category/2216585)改编而来。具体来说，在[joveth](http://blog.csdn.net/joveth)教程的基础上，实验楼修改了教程以适配最新版本的`Revel`。

为了能正常开始实验，我们需要设置 `GOPATH` 环境变量以及启动mongodb，每次开始实验前，请执行以下命令：

```
$ cd /home/shiyanlou/revel_mgo_blog
$ export GOPATH=`pwd`
$ export PATH=$PATH:$GOPATH/bin
$ sudo mongod --fork -f /etc/mongodb.conf
```

## 归档功能

我们实现的归档功能将按照年份来归档。

还是按照之前实现其他功能的步骤，先实现model，然后实现controller，接着实现页面，添加路由这样一个功能就完成啦。下面就让我们实现归档功能。

首先创建`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/models/history.go`文件，输入以下内容：

```
package models
import (
  "github.com/revel/revel"
  "gopkg.in/mgo.v2/bson"
  "time"
)
type History struct {
  Year int
  Blogs []Blog
}

// 插入归档
func (dao *Dao) InsertHistory(history *History) error {
  historyCollection := dao.session.DB(DbName).C(HistoryCollection)
  err := historyCollection.Insert(history)
  if err != nil {
    revel.WARN.Printf("Unable to save History: %v error %v", history, err)
  }
  return err
}

// 查找归档
func (dao *Dao) FindHistory() []History{
  historyCollection := dao.session.DB(DbName).C(HistoryCollection)
  his := []History{}
  query := historyCollection.Find(bson.M{}).Sort("-year")
  query.All(&his)
  return his
}

// 删除归档
func (dao *Dao) RemoveAll() error{
  historyCollection := dao.session.DB(DbName).C(HistoryCollection)
  _,err := historyCollection.RemoveAll(bson.M{})
  if err != nil {
    revel.WARN.Printf("Unable to RemoveAll: error %v",  err)
  }
  return err
}

// 创建所有归档
func (dao *Dao) CreateAllHistory() {
  dao.RemoveAll();
  var end int = time.Now().Year();
  for i:=BaseYear;i<=end;i++{
    history := new(History);
    history.Year = i;
    dao.InsertHistory(history);
  }
}
```

然后我们实现相应的controller，编辑`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go`，添加以下方法：

```
func (c App) History() revel.Result {
  dao, err := models.NewDao()
  if err != nil {
    c.Response.Status = 500
    return c.RenderError(err)
  }
  defer dao.Close()
  dao.CreateAllHistory();
  historys := dao.FindHistory();
  for i,_ := range historys{
    historys[i].Blogs =dao.FindBlogsByYear(historys[i].Year); 
  }
  return c.Render(historys)
}

```

可以看到我们用到了一个新方法`FindBlogsByYear`，下面我们编辑`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/models/blog.go`添加该方法：

```
func (dao *Dao) FindBlogsByYear(year int) []Blog{
  blogCollection := dao.session.DB(DbName).C(BlogCollection)
  blogs := []Blog{}
  query := blogCollection.Find(bson.M{"year":year}).Sort("-cdate")
  query.All(&blogs)
  return blogs
}

```

然后我们添加`History.html`页面，用于显示归档。创建文件`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/History.html`, 输入以下内容：

```
 {{set . "title" "History blogs - GBlog"}}
  {{set . "history" "active" }}
  {{template "header.html" .}}
  <div class="content">
      <div class="history-nav" id="his">
         <div class="history-title">
           File
         </div>
         {{if .historys}}
         {{range $index,$history :=.historys}}
         <div class="history-cell">
            <div class="panel-heading" style="padding:0;border-bottom: 1px dashed #ccc;">
              <a data-toggle="collapse" data-toggle="collapse" data-parent="#his" href="#collapseOne{{$index}}">{{$history.Year}}</a>
            </div>
            <div id="collapseOne{{$index}}" class="panel-collapse collapse in">
              <div class="panel-body" style="padding:0 20px;">
                <ul style="padding:10px 10px;list-style:none;">
                  {{if $history.Blogs }}
                  {{range $blog :=$history.Blogs}}
                  <li><time>{{$blog.CDate.Format "2006-01-02"}}</time><a href="#">{{$blog.GetShortTitle }}</a><span class="history-auth">By {{$blog.Email}}</span></li>
                  {{end}}
                  {{end}}
                </ul>
              </div>
            </div>
         </div>
         {{end}}
         {{end}}
      </div>
  </div>
  {{template "footer.html" .}}
```

最后，我们添加相应的路由，编辑`/home/shiyanlou/revel_mgo_blog/src/GBlog/conf/routes`文件，添加以下路由：

```
GET     /history                                App.History
```

到这里，整个归档功能就实现了，效果如下：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1417timestamp1445416174607.png/wm)



## 提醒功能

在之前的设计中，博客菜单栏最左边有红色的提醒标志，是怎么实现的呢？ 它的作用是提醒最近一小时内的更新，所以我们应该有一个可以根据时间查询博客文章的方法。编辑`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/models/blog.go`文件，添加以下方法：

```
func (dao *Dao) FindBlogsByDate(start time.Time) int{
  blogCollection := dao.session.DB(DbName).C(BlogCollection)
  query := blogCollection.Find(bson.M{"cdate":bson.M{"$gte": start}})
  cnt,err := query.Count();
  if err!=nil{
    revel.WARN.Printf("Unable to Count blog: error %v", err)
  }
  return cnt
}

```

我们这里使用`$gte`高级查询，具体可以查看[mgo的文档](https://godoc.org/gopkg.in/mgo.v2)。

接着我们修改`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go`中的`Index`方法为以下内容：

```
func (c App) Index() revel.Result {
  dao, err := models.NewDao()
  if err != nil {
    c.Response.Status = 500
    return c.RenderError(err)
  }
  defer dao.Close()
  //dao := models.NewDao(c.MongoSession)
  blogs := dao.FindBlogs()
  now := time.Now().Add(-1 * time.Hour)
  recentCnt :=dao.FindBlogsByDate(now);
  return c.Render(blogs,recentCnt)
}

```

以上代码相比以前增加了查询最近一小时更新文章数量的代码。我们在引用了`time`模块，所以不要忘记在`app.go`中导入`time`模块。

接着，我们在模板中显示传递的`recentCnt`参数。修改`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/header.html`中的代码块：

```
    <li class="{{.home}}">
                <a href="/" >Home</a>
                <span class="navbar-news " title="There is an update in the last 1 hours">1</span>
    </li>
```

为以下内容：

```
    <li class="{{.home}}">
            <a href="/" >Home</a>
            {{if .recentCnt}}
            {{if gt .recentCnt 0}}
            <span class="navbar-news " title="There is {{.recentCnt}} update in the last 1 hours">{{.recentCnt}}</span>
            {{end}}
            {{end}}
          </li>
```

再次刷新页面，就可以看到效果啦，如果没有效果，点击投稿，再次发布一篇文章就可以看到效果了。

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1417timestamp1445418435192.png/wm)


## about 页面

经过之前的学习，相信添加about页面的工作已经信手拈来啦。创建文件`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/views/App/About.html`，输入以下内容：

```
{{set . "title" "About me - GBlog"}}
{{set . "about" "active" }}
{{template "header.html" .}}
<div class="content">
    <div class="history-nav" id="his">
       <div class="history-title">
       About the course
       </div>
       <div class="history-cell">
        <p>
        <a href="https://www.shiyanlou.com">shiyanlou's</a>Course《Based on Revel and MgO's blog.》,The course reference：<a href="http://blog.csdn.net/joveth">joveth's</a>Article<a href="http://blog.csdn.net/jov123/article/category/2216585"> </a
        </p>
       </div>
    </div>
</div>
{{template "footer.html" .}}
```

编辑文件`/home/shiyanlou/revel_mgo_blog/src/GBlog/app/controllers/app.go`，添加以下方法：

```
func (c App) revel.Result {
    return c.Render()
}
```

接着添加路由，编辑文件: ``，添加以下路由：

```
GET    /about                                   App.About
```

现在让我们看下效果：

![此处输入图片的描述](https://dn-anything-about-doc.qbox.me/document-uid5348labid1417timestamp1445419981479.png/wm)


到目前为止整个课程就完结了。但是我们还有一些功能没有实现，比如菜单栏中的`Email`和`RSS`功能，这些功能我们将作为练习，由你自己完成。


## 练习  

+ 在于[joveth](http://blog.csdn.net/joveth)原来的教程中，还有一个email墙功能，你能参考目前我们实现的其他功能实现它吗？ 提示：你需要实现一个`Email`model用于存储邮箱地址，一个controller和相应的页面用于显示email地址。同时email地址，可以在发布文章时，评论时和留言的时候收集。