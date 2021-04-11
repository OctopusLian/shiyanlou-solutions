<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />
《实验楼Sql注入示例》<br>
<a href="creattable.php"><input type="button" value="创建表单"></input></a><br>
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
$sql = 'CREATE DATABASE SQLINJECT';
$retval = $conn->query($sql);
if(! $retval )
{
 die('Could not create database: ' . mysqli_error());
}
echo "Database SQLINJECT created successfully\n";
mysqli_close($conn);
?>
</body>
</html>
