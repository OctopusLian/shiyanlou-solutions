CREATE TABLE IF NOT EXISTS `users`(
   `id` INT UNSIGNED AUTO_INCREMENT,
   `username` VARCHAR(100) NOT NULL,
   `password` VARCHAR(40) NOT NULL,
   `submission_date` DATE,
   PRIMARY KEY ( `id` )
)ENGINE=InnoDB DEFAULT CHARSET=utf8;
