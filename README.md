1、修改setting 文件 mysql、ES、Milvus 配置

2、终端执行：

```
python manage.py makemigrations
python manage.py migrate
```

3、数据库执行

```
ALTER TABLE `auth_user`
MODIFY `username` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL;

CREATE TABLE `KNOW_USER_INFO` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '用户id',
  `username` varchar(150) NOT NULL COMMENT '用户名',
  `database_type` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL COMMENT '知识库类型',
  `index_name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL COMMENT '知识库名称',
  `index_statute` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL COMMENT '知识库状态（执行中，已完成）',
  `embedding` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL COMMENT '算法名称',
  `get_date` datetime DEFAULT NULL COMMENT '创建日期',
  `kg_count` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL COMMENT '数量',
  `project_name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL COMMENT '关联项目名称',
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_know_user_info_username` FOREIGN KEY (`username`) REFERENCES `auth_user` (`username`)
    ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
```

