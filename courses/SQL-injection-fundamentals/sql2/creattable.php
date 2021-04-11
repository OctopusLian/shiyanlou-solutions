<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />
《实验楼Sql注入示例》<br>
<a href="creatdata.php"><input type="button" value="初始化数据"></input></a><br>
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
$sql2 = "CREATE TABLE users( ".
        "id INT NOT NULL AUTO_INCREMENT, ".
        "username VARCHAR(100) NOT NULL, ".
        "password VARCHAR(40) NOT NULL, ".
        "submission_date DATE, ".
        "PRIMARY KEY ( id ))ENGINE=InnoDB DEFAULT CHARSET=utf8; ";

$retval2 =  $conn->query($sql2);
if(! $retval2 )
{
 die('Could not create table: ' . mysqli_error());
}
echo "Table users created successfully\n";
mysqli_close($conn);
?>
<br>
</body>
</html>

