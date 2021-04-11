<html>
<head>
<title>登录验证</title>
<meta http-equiv="content-type" content="text/html;charset=utf-8">
</head>

<body>
<?php

       $conn=@mysqli_connect("localhost",'root','') or die("数据库连接失败！");;
       mysqli_select_db($conn, 'SQLINJECT') or die("您要选择的数据库不存在");

       $name=$_POST['username'];
       $pwd=$_POST['password'];
       $sql="select * from users where username='$name' and password='$pwd'";

       $query= $conn->query($sql);
       $arr=mysqli_fetch_array($query);
       if(is_array($arr)){
              header("Location:success.php");
       }else{
              echo "您的用户名或密码输入有误，<a href=\"index.html\">请重新登录！</a>";
       }

?>
</body>
</html>
