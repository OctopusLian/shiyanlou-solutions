<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />
《实验楼Sql注入示例》<br>
<a href="index.html"><input type="button" value="返回首页"></input></a><br>
<title>Creating MySQL Database</title>
</head>
<body>
<?php
$dbhost = 'localhost:3306';
$dbuser = 'root';
$dbpass = '';
$conn = mysqli_connect($dbhost, $dbuser, $dbpass);
if(! $conn )
{
 die('Could not connect: ' . mysqli_error());
}
echo 'Connected successfully<br />';
mysqli_select_db($conn, 'SQLINJECT') or die("您要选择的数据库不存在");
$sql3 = "INSERT INTO users (username,password)".
	"VALUES('shiyanlou','shiyanloupd');";

$retval3 = $conn->query($sql3);
if(! $retval3 )
{
 die('Could not Initialized Data: ' . mysqli_error());
}
echo "Initialized Data successfully\n";
mysqli_close($conn);
?>
<br>
</body>
</html>

