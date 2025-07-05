-- MySQL dump 10.13  Distrib 8.0.42, for Linux (x86_64)
--
-- Host: 103.6.171.147    Database: mlflow_db
-- ------------------------------------------------------
-- Server version	8.4.5

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `alembic_version`
--

DROP TABLE IF EXISTS `alembic_version`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `alembic_version` (
  `version_num` varchar(32) NOT NULL,
  PRIMARY KEY (`version_num`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `alembic_version`
--

LOCK TABLES `alembic_version` WRITE;
/*!40000 ALTER TABLE `alembic_version` DISABLE KEYS */;
INSERT INTO `alembic_version` VALUES ('0584bdc529eb');
/*!40000 ALTER TABLE `alembic_version` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `datasets`
--

DROP TABLE IF EXISTS `datasets`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `datasets` (
  `dataset_uuid` varchar(36) NOT NULL,
  `experiment_id` int NOT NULL,
  `name` varchar(500) NOT NULL,
  `digest` varchar(36) NOT NULL,
  `dataset_source_type` varchar(36) NOT NULL,
  `dataset_source` text NOT NULL,
  `dataset_schema` mediumtext,
  `dataset_profile` mediumtext,
  PRIMARY KEY (`experiment_id`,`name`,`digest`),
  KEY `index_datasets_dataset_uuid` (`dataset_uuid`),
  KEY `index_datasets_experiment_id_dataset_source_type` (`experiment_id`,`dataset_source_type`),
  CONSTRAINT `fk_datasets_experiment_id_experiments` FOREIGN KEY (`experiment_id`) REFERENCES `experiments` (`experiment_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `datasets`
--

LOCK TABLES `datasets` WRITE;
/*!40000 ALTER TABLE `datasets` DISABLE KEYS */;
INSERT INTO `datasets` VALUES ('541ce723c97d40a6811a20176457fae2',1,'dataset','5ce4c568','code','{\"tags\": {\"mlflow.user\": \"root\", \"mlflow.source.name\": \"./work/notebook_experiments/LightGBM_StrokePrediction_MLflow.py\", \"mlflow.source.type\": \"LOCAL\", \"mlflow.source.git.commit\": \"625b88275c0ffef94223ef15a54eb83d6ed68c15\"}}','{\"mlflow_tensorspec\": {\"features\": \"[{\\\"type\\\": \\\"tensor\\\", \\\"tensor-spec\\\": {\\\"dtype\\\": \\\"float64\\\", \\\"shape\\\": [-1, 25]}}]\", \"targets\": null}}','{\"features_shape\": [1022, 25], \"features_size\": 25550, \"features_nbytes\": 204400}'),('d734f5c8d4534470b8bebc0c4a61d930',1,'dataset','648447b8','code','{\"tags\": {\"mlflow.user\": \"root\", \"mlflow.source.name\": \"./work/notebook_experiments/LightGBM_StrokePrediction_MLflow.py\", \"mlflow.source.type\": \"LOCAL\", \"mlflow.source.git.commit\": \"625b88275c0ffef94223ef15a54eb83d6ed68c15\"}}','{\"mlflow_tensorspec\": {\"features\": \"[{\\\"type\\\": \\\"tensor\\\", \\\"tensor-spec\\\": {\\\"dtype\\\": \\\"float64\\\", \\\"shape\\\": [-1, 25]}}]\", \"targets\": \"[{\\\"type\\\": \\\"tensor\\\", \\\"tensor-spec\\\": {\\\"dtype\\\": \\\"int64\\\", \\\"shape\\\": [-1]}}]\"}}','{\"features_shape\": [7776, 25], \"features_size\": 194400, \"features_nbytes\": 1555200, \"targets_shape\": [7776], \"targets_size\": 7776, \"targets_nbytes\": 62208}'),('fc58759809434f41aa6cd3a7005f8890',3,'dataset','d4b23cb1','code','{\"tags\": {\"mlflow.user\": \"root\", \"mlflow.source.name\": \"./work/notebook_experiments/XGBoost_StrokePrediction_MLflow.py\", \"mlflow.source.type\": \"LOCAL\", \"mlflow.source.git.commit\": \"625b88275c0ffef94223ef15a54eb83d6ed68c15\"}}','{\"mlflow_tensorspec\": {\"features\": \"[{\\\"type\\\": \\\"tensor\\\", \\\"tensor-spec\\\": {\\\"dtype\\\": \\\"float32\\\", \\\"shape\\\": [-1, 23]}}]\", \"targets\": null}}','{\"features_shape\": [7776, 23], \"features_size\": 178848, \"features_nbytes\": 715392}'),('0cc160bbd2534c5faf1387bd7213a516',3,'dataset','f898e53a','code','{\"tags\": {\"mlflow.user\": \"root\", \"mlflow.source.name\": \"./work/notebook_experiments/XGBoost_StrokePrediction_MLflow.py\", \"mlflow.source.type\": \"LOCAL\", \"mlflow.source.git.commit\": \"625b88275c0ffef94223ef15a54eb83d6ed68c15\"}}','{\"mlflow_tensorspec\": {\"features\": \"[{\\\"type\\\": \\\"tensor\\\", \\\"tensor-spec\\\": {\\\"dtype\\\": \\\"float64\\\", \\\"shape\\\": [-1, 23]}}]\", \"targets\": null}}','{\"features_shape\": [1022, 23], \"features_size\": 23506, \"features_nbytes\": 188048}');
/*!40000 ALTER TABLE `datasets` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `experiment_tags`
--

DROP TABLE IF EXISTS `experiment_tags`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `experiment_tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(5000) DEFAULT NULL,
  `experiment_id` int NOT NULL,
  PRIMARY KEY (`key`,`experiment_id`),
  KEY `experiment_id` (`experiment_id`),
  CONSTRAINT `experiment_tags_ibfk_1` FOREIGN KEY (`experiment_id`) REFERENCES `experiments` (`experiment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `experiment_tags`
--

LOCK TABLES `experiment_tags` WRITE;
/*!40000 ALTER TABLE `experiment_tags` DISABLE KEYS */;
/*!40000 ALTER TABLE `experiment_tags` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `experiments`
--

DROP TABLE IF EXISTS `experiments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `experiments` (
  `experiment_id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(256) NOT NULL,
  `artifact_location` varchar(256) DEFAULT NULL,
  `lifecycle_stage` varchar(32) DEFAULT NULL,
  `creation_time` bigint DEFAULT NULL,
  `last_update_time` bigint DEFAULT NULL,
  PRIMARY KEY (`experiment_id`),
  UNIQUE KEY `name` (`name`),
  CONSTRAINT `experiments_lifecycle_stage` CHECK ((`lifecycle_stage` in (_utf8mb4'active',_utf8mb4'deleted')))
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `experiments`
--

LOCK TABLES `experiments` WRITE;
/*!40000 ALTER TABLE `experiments` DISABLE KEYS */;
INSERT INTO `experiments` VALUES (0,'Default','file:///mlruns/0','active',1751569630008,1751569630008),(1,'Stroke_Prediction_LightGBM','file:///mlruns/1','active',1751570131982,1751570131982),(2,'Stroke_Prediction_RandomForest','file:///mlruns/2','active',1751570188395,1751570188395),(3,'Stroke_Prediction_XGBoost','file:///mlruns/3','active',1751570218305,1751570218305),(4,'Stroke_Prediction_LogisticRegression','file:///mlruns/4','active',1751570265348,1751570265348),(5,'Stroke_Prediction_SVM','file:///mlruns/5','active',1751570401335,1751570401335);
/*!40000 ALTER TABLE `experiments` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `input_tags`
--

DROP TABLE IF EXISTS `input_tags`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `input_tags` (
  `input_uuid` varchar(36) NOT NULL,
  `name` varchar(255) NOT NULL,
  `value` varchar(500) NOT NULL,
  PRIMARY KEY (`input_uuid`,`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `input_tags`
--

LOCK TABLES `input_tags` WRITE;
/*!40000 ALTER TABLE `input_tags` DISABLE KEYS */;
INSERT INTO `input_tags` VALUES ('135fe10061d448c4a3b314228eeae0a4','mlflow.data.context','eval'),('42bead4babf04a7b97b2a56d181b73f7','mlflow.data.context','train'),('5dc07911bf99446cb5b4ac815a21bc5b','mlflow.data.context','eval'),('8a7bf49507c743cfac0e1385cf2e28f6','mlflow.data.context','train');
/*!40000 ALTER TABLE `input_tags` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `inputs`
--

DROP TABLE IF EXISTS `inputs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `inputs` (
  `input_uuid` varchar(36) NOT NULL,
  `source_type` varchar(36) NOT NULL,
  `source_id` varchar(36) NOT NULL,
  `destination_type` varchar(36) NOT NULL,
  `destination_id` varchar(36) NOT NULL,
  PRIMARY KEY (`source_type`,`source_id`,`destination_type`,`destination_id`),
  KEY `index_inputs_destination_type_destination_id_source_type` (`destination_type`,`destination_id`,`source_type`),
  KEY `index_inputs_input_uuid` (`input_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `inputs`
--

LOCK TABLES `inputs` WRITE;
/*!40000 ALTER TABLE `inputs` DISABLE KEYS */;
INSERT INTO `inputs` VALUES ('135fe10061d448c4a3b314228eeae0a4','DATASET','0cc160bbd2534c5faf1387bd7213a516','RUN','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('42bead4babf04a7b97b2a56d181b73f7','DATASET','d734f5c8d4534470b8bebc0c4a61d930','RUN','c552b07b144b40a3930a2024878208f7'),('5dc07911bf99446cb5b4ac815a21bc5b','DATASET','541ce723c97d40a6811a20176457fae2','RUN','c552b07b144b40a3930a2024878208f7'),('8a7bf49507c743cfac0e1385cf2e28f6','DATASET','fc58759809434f41aa6cd3a7005f8890','RUN','e4d6ee7c50cc4dbeb8cf2689608bd94a');
/*!40000 ALTER TABLE `inputs` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `latest_metrics`
--

DROP TABLE IF EXISTS `latest_metrics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `latest_metrics` (
  `key` varchar(250) NOT NULL,
  `value` double NOT NULL,
  `timestamp` bigint DEFAULT NULL,
  `step` bigint NOT NULL,
  `is_nan` tinyint(1) NOT NULL,
  `run_uuid` varchar(32) NOT NULL,
  PRIMARY KEY (`key`,`run_uuid`),
  KEY `index_latest_metrics_run_uuid` (`run_uuid`),
  CONSTRAINT `latest_metrics_ibfk_1` FOREIGN KEY (`run_uuid`) REFERENCES `runs` (`run_uuid`),
  CONSTRAINT `latest_metrics_chk_1` CHECK ((`is_nan` in (0,1)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `latest_metrics`
--

LOCK TABLES `latest_metrics` WRITE;
/*!40000 ALTER TABLE `latest_metrics` DISABLE KEYS */;
INSERT INTO `latest_metrics` VALUES ('accuracy',0.8493150684931506,1751570189737,0,0,'86f3f7c6a2a54692bafb592fd871a6b3'),('accuracy',0.5645792563600783,1751570311702,0,0,'ad3797b88917440ba31793ba47c5fc52'),('accuracy',0.7544031311154599,1751570402966,0,0,'b3cf3b20ad7d4285b3a79f668007e67f'),('accuracy',0.7495107632093934,1751570136581,0,0,'c552b07b144b40a3930a2024878208f7'),('accuracy',0.662426614481409,1751570222105,0,0,'e4d6ee7c50cc4dbeb8cf2689608bd94a'),('f1',0.19292604501607716,1751570402966,0,0,'b3cf3b20ad7d4285b3a79f668007e67f'),('f1',0.16883116883116883,1751570136581,0,0,'c552b07b144b40a3930a2024878208f7'),('f1_score',0.2222222222222222,1751570189766,0,0,'86f3f7c6a2a54692bafb592fd871a6b3'),('f1_score',0.20677361853832443,1751570311732,0,0,'ad3797b88917440ba31793ba47c5fc52'),('f1_score',0.18439716312056736,1751570222105,0,0,'e4d6ee7c50cc4dbeb8cf2689608bd94a'),('pr_auc',0.12324327794704645,1751570402966,0,0,'b3cf3b20ad7d4285b3a79f668007e67f'),('pr_auc',0.12274154926185658,1751570136581,0,0,'c552b07b144b40a3930a2024878208f7'),('pr_auc',0.14847421621102852,1751570222105,0,0,'e4d6ee7c50cc4dbeb8cf2689608bd94a'),('precision',0.16176470588235295,1751570189747,0,0,'86f3f7c6a2a54692bafb592fd871a6b3'),('precision',0.11623246492985972,1751570311713,0,0,'ad3797b88917440ba31793ba47c5fc52'),('precision',0.11494252873563218,1751570402966,0,0,'b3cf3b20ad7d4285b3a79f668007e67f'),('precision',0.10077519379844961,1751570136581,0,0,'c552b07b144b40a3930a2024878208f7'),('precision',0.10455764075067024,1751570222105,0,0,'e4d6ee7c50cc4dbeb8cf2689608bd94a'),('recall',0.3548387096774194,1751570189756,0,0,'86f3f7c6a2a54692bafb592fd871a6b3'),('recall',0.9354838709677419,1751570311723,0,0,'ad3797b88917440ba31793ba47c5fc52'),('recall',0.6,1751570402966,0,0,'b3cf3b20ad7d4285b3a79f668007e67f'),('recall',0.52,1751570136581,0,0,'c552b07b144b40a3930a2024878208f7'),('recall',0.78,1751570222105,0,0,'e4d6ee7c50cc4dbeb8cf2689608bd94a'),('roc_auc',0.8067540322580645,1751570189774,0,0,'86f3f7c6a2a54692bafb592fd871a6b3'),('roc_auc',0.847513440860215,1751570311742,0,0,'ad3797b88917440ba31793ba47c5fc52'),('roc_auc',0.7661111111111112,1751570402966,0,0,'b3cf3b20ad7d4285b3a79f668007e67f'),('roc_auc',0.7658641975308642,1751570136581,0,0,'c552b07b144b40a3930a2024878208f7'),('roc_auc',0.7832613168724281,1751570222105,0,0,'e4d6ee7c50cc4dbeb8cf2689608bd94a');
/*!40000 ALTER TABLE `latest_metrics` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `metrics`
--

DROP TABLE IF EXISTS `metrics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `metrics` (
  `key` varchar(250) NOT NULL,
  `value` double NOT NULL,
  `timestamp` bigint NOT NULL,
  `run_uuid` varchar(32) NOT NULL,
  `step` bigint NOT NULL DEFAULT '0',
  `is_nan` tinyint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`key`,`timestamp`,`step`,`run_uuid`,`value`,`is_nan`),
  KEY `index_metrics_run_uuid` (`run_uuid`),
  CONSTRAINT `metrics_ibfk_1` FOREIGN KEY (`run_uuid`) REFERENCES `runs` (`run_uuid`),
  CONSTRAINT `metrics_chk_1` CHECK ((`is_nan` in (0,1))),
  CONSTRAINT `metrics_chk_2` CHECK ((`is_nan` in (0,1)))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `metrics`
--

LOCK TABLES `metrics` WRITE;
/*!40000 ALTER TABLE `metrics` DISABLE KEYS */;
INSERT INTO `metrics` VALUES ('accuracy',0.8493150684931506,1751570189737,'86f3f7c6a2a54692bafb592fd871a6b3',0,0),('f1_score',0.2222222222222222,1751570189766,'86f3f7c6a2a54692bafb592fd871a6b3',0,0),('precision',0.16176470588235295,1751570189747,'86f3f7c6a2a54692bafb592fd871a6b3',0,0),('recall',0.3548387096774194,1751570189756,'86f3f7c6a2a54692bafb592fd871a6b3',0,0),('roc_auc',0.8067540322580645,1751570189774,'86f3f7c6a2a54692bafb592fd871a6b3',0,0),('accuracy',0.5645792563600783,1751570311702,'ad3797b88917440ba31793ba47c5fc52',0,0),('f1_score',0.20677361853832443,1751570311732,'ad3797b88917440ba31793ba47c5fc52',0,0),('precision',0.11623246492985972,1751570311713,'ad3797b88917440ba31793ba47c5fc52',0,0),('recall',0.9354838709677419,1751570311723,'ad3797b88917440ba31793ba47c5fc52',0,0),('roc_auc',0.847513440860215,1751570311742,'ad3797b88917440ba31793ba47c5fc52',0,0),('accuracy',0.7544031311154599,1751570402966,'b3cf3b20ad7d4285b3a79f668007e67f',0,0),('f1',0.19292604501607716,1751570402966,'b3cf3b20ad7d4285b3a79f668007e67f',0,0),('pr_auc',0.12324327794704645,1751570402966,'b3cf3b20ad7d4285b3a79f668007e67f',0,0),('precision',0.11494252873563218,1751570402966,'b3cf3b20ad7d4285b3a79f668007e67f',0,0),('recall',0.6,1751570402966,'b3cf3b20ad7d4285b3a79f668007e67f',0,0),('roc_auc',0.7661111111111112,1751570402966,'b3cf3b20ad7d4285b3a79f668007e67f',0,0),('accuracy',0.7495107632093934,1751570136581,'c552b07b144b40a3930a2024878208f7',0,0),('f1',0.16883116883116883,1751570136581,'c552b07b144b40a3930a2024878208f7',0,0),('pr_auc',0.12274154926185658,1751570136581,'c552b07b144b40a3930a2024878208f7',0,0),('precision',0.10077519379844961,1751570136581,'c552b07b144b40a3930a2024878208f7',0,0),('recall',0.52,1751570136581,'c552b07b144b40a3930a2024878208f7',0,0),('roc_auc',0.7658641975308642,1751570136581,'c552b07b144b40a3930a2024878208f7',0,0),('accuracy',0.662426614481409,1751570222105,'e4d6ee7c50cc4dbeb8cf2689608bd94a',0,0),('f1_score',0.18439716312056736,1751570222105,'e4d6ee7c50cc4dbeb8cf2689608bd94a',0,0),('pr_auc',0.14847421621102852,1751570222105,'e4d6ee7c50cc4dbeb8cf2689608bd94a',0,0),('precision',0.10455764075067024,1751570222105,'e4d6ee7c50cc4dbeb8cf2689608bd94a',0,0),('recall',0.78,1751570222105,'e4d6ee7c50cc4dbeb8cf2689608bd94a',0,0),('roc_auc',0.7832613168724281,1751570222105,'e4d6ee7c50cc4dbeb8cf2689608bd94a',0,0);
/*!40000 ALTER TABLE `metrics` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `model_version_tags`
--

DROP TABLE IF EXISTS `model_version_tags`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `model_version_tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(5000) DEFAULT NULL,
  `name` varchar(256) NOT NULL,
  `version` int NOT NULL,
  PRIMARY KEY (`key`,`name`,`version`),
  KEY `name` (`name`,`version`),
  CONSTRAINT `model_version_tags_ibfk_1` FOREIGN KEY (`name`, `version`) REFERENCES `model_versions` (`name`, `version`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `model_version_tags`
--

LOCK TABLES `model_version_tags` WRITE;
/*!40000 ALTER TABLE `model_version_tags` DISABLE KEYS */;
/*!40000 ALTER TABLE `model_version_tags` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `model_versions`
--

DROP TABLE IF EXISTS `model_versions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `model_versions` (
  `name` varchar(256) NOT NULL,
  `version` int NOT NULL,
  `creation_time` bigint DEFAULT NULL,
  `last_updated_time` bigint DEFAULT NULL,
  `description` varchar(5000) DEFAULT NULL,
  `user_id` varchar(256) DEFAULT NULL,
  `current_stage` varchar(20) DEFAULT NULL,
  `source` varchar(500) DEFAULT NULL,
  `run_id` varchar(32) DEFAULT NULL,
  `status` varchar(20) DEFAULT NULL,
  `status_message` varchar(500) DEFAULT NULL,
  `run_link` varchar(500) DEFAULT NULL,
  `storage_location` varchar(500) DEFAULT NULL,
  PRIMARY KEY (`name`,`version`),
  CONSTRAINT `model_versions_ibfk_1` FOREIGN KEY (`name`) REFERENCES `registered_models` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `model_versions`
--

LOCK TABLES `model_versions` WRITE;
/*!40000 ALTER TABLE `model_versions` DISABLE KEYS */;
/*!40000 ALTER TABLE `model_versions` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `params`
--

DROP TABLE IF EXISTS `params`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `params` (
  `key` varchar(250) NOT NULL,
  `value` varchar(8000) NOT NULL,
  `run_uuid` varchar(32) NOT NULL,
  PRIMARY KEY (`key`,`run_uuid`),
  KEY `index_params_run_uuid` (`run_uuid`),
  CONSTRAINT `params_ibfk_1` FOREIGN KEY (`run_uuid`) REFERENCES `runs` (`run_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `params`
--

LOCK TABLES `params` WRITE;
/*!40000 ALTER TABLE `params` DISABLE KEYS */;
INSERT INTO `params` VALUES ('base_score','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('booster','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('boosting_type','gbdt','c552b07b144b40a3930a2024878208f7'),('C','1.0','b3cf3b20ad7d4285b3a79f668007e67f'),('class_weight','{1: 19.537688442211056, 0: 1.0}','b3cf3b20ad7d4285b3a79f668007e67f'),('classifier__C','0.01','ad3797b88917440ba31793ba47c5fc52'),('classifier__max_iter','500','ad3797b88917440ba31793ba47c5fc52'),('classifier__penalty','l1','ad3797b88917440ba31793ba47c5fc52'),('classifier__solver','liblinear','ad3797b88917440ba31793ba47c5fc52'),('colsample_bylevel','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('colsample_bynode','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('colsample_bytree','1.0','c552b07b144b40a3930a2024878208f7'),('colsample_bytree','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('custom_metric','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('device','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('early_stopping_rounds','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('eval_metric','aucpr','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('gamma','scale','b3cf3b20ad7d4285b3a79f668007e67f'),('gamma','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('grow_policy','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('interaction_constraints','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('keep_training_booster','False','c552b07b144b40a3930a2024878208f7'),('kernel','rbf','b3cf3b20ad7d4285b3a79f668007e67f'),('learning_rate','0.05','c552b07b144b40a3930a2024878208f7'),('learning_rate','0.1','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('max_bin','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('max_cat_threshold','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('max_cat_to_onehot','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('max_delta_step','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('max_depth','10','86f3f7c6a2a54692bafb592fd871a6b3'),('max_depth','5','c552b07b144b40a3930a2024878208f7'),('max_depth','3','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('max_leaves','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('maximize','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('metric','[\'aucpr\']','c552b07b144b40a3930a2024878208f7'),('min_child_samples','20','c552b07b144b40a3930a2024878208f7'),('min_child_weight','0.001','c552b07b144b40a3930a2024878208f7'),('min_child_weight','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('min_samples_split','5','86f3f7c6a2a54692bafb592fd871a6b3'),('min_split_gain','0.0','c552b07b144b40a3930a2024878208f7'),('monotone_constraints','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('multi_strategy','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('n_estimators','200','86f3f7c6a2a54692bafb592fd871a6b3'),('n_estimators','200','c552b07b144b40a3930a2024878208f7'),('n_estimators','100','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('n_jobs','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('num_boost_round','200','c552b07b144b40a3930a2024878208f7'),('num_boost_round','100','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('num_leaves','31','c552b07b144b40a3930a2024878208f7'),('num_parallel_tree','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('num_threads','2','c552b07b144b40a3930a2024878208f7'),('objective','binary','c552b07b144b40a3930a2024878208f7'),('objective','binary:logistic','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('random_state','42','c552b07b144b40a3930a2024878208f7'),('random_state','42','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('reg_alpha','0.0','c552b07b144b40a3930a2024878208f7'),('reg_alpha','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('reg_lambda','0.0','c552b07b144b40a3930a2024878208f7'),('reg_lambda','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('sampling_method','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('scale_pos_weight','19.537688442211056','c552b07b144b40a3930a2024878208f7'),('scale_pos_weight','19.537688442211056','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('subsample','1.0','c552b07b144b40a3930a2024878208f7'),('subsample','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('subsample_for_bin','200000','c552b07b144b40a3930a2024878208f7'),('subsample_freq','0','c552b07b144b40a3930a2024878208f7'),('threshold','0.3','ad3797b88917440ba31793ba47c5fc52'),('tree_method','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('use_label_encoder','False','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('validate_parameters','None','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('verbose_eval','True','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('verbosity','-1','c552b07b144b40a3930a2024878208f7'),('verbosity','None','e4d6ee7c50cc4dbeb8cf2689608bd94a');
/*!40000 ALTER TABLE `params` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `registered_model_aliases`
--

DROP TABLE IF EXISTS `registered_model_aliases`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `registered_model_aliases` (
  `alias` varchar(256) NOT NULL,
  `version` int NOT NULL,
  `name` varchar(256) NOT NULL,
  PRIMARY KEY (`name`,`alias`),
  CONSTRAINT `registered_model_alias_name_fkey` FOREIGN KEY (`name`) REFERENCES `registered_models` (`name`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `registered_model_aliases`
--

LOCK TABLES `registered_model_aliases` WRITE;
/*!40000 ALTER TABLE `registered_model_aliases` DISABLE KEYS */;
/*!40000 ALTER TABLE `registered_model_aliases` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `registered_model_tags`
--

DROP TABLE IF EXISTS `registered_model_tags`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `registered_model_tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(5000) DEFAULT NULL,
  `name` varchar(256) NOT NULL,
  PRIMARY KEY (`key`,`name`),
  KEY `name` (`name`),
  CONSTRAINT `registered_model_tags_ibfk_1` FOREIGN KEY (`name`) REFERENCES `registered_models` (`name`) ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `registered_model_tags`
--

LOCK TABLES `registered_model_tags` WRITE;
/*!40000 ALTER TABLE `registered_model_tags` DISABLE KEYS */;
/*!40000 ALTER TABLE `registered_model_tags` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `registered_models`
--

DROP TABLE IF EXISTS `registered_models`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `registered_models` (
  `name` varchar(256) NOT NULL,
  `creation_time` bigint DEFAULT NULL,
  `last_updated_time` bigint DEFAULT NULL,
  `description` varchar(5000) DEFAULT NULL,
  PRIMARY KEY (`name`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `registered_models`
--

LOCK TABLES `registered_models` WRITE;
/*!40000 ALTER TABLE `registered_models` DISABLE KEYS */;
/*!40000 ALTER TABLE `registered_models` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `runs`
--

DROP TABLE IF EXISTS `runs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `runs` (
  `run_uuid` varchar(32) NOT NULL,
  `name` varchar(250) DEFAULT NULL,
  `source_type` varchar(20) DEFAULT NULL,
  `source_name` varchar(500) DEFAULT NULL,
  `entry_point_name` varchar(50) DEFAULT NULL,
  `user_id` varchar(256) DEFAULT NULL,
  `status` varchar(9) DEFAULT NULL,
  `start_time` bigint DEFAULT NULL,
  `end_time` bigint DEFAULT NULL,
  `source_version` varchar(50) DEFAULT NULL,
  `lifecycle_stage` varchar(20) DEFAULT NULL,
  `artifact_uri` varchar(200) DEFAULT NULL,
  `experiment_id` int DEFAULT NULL,
  `deleted_time` bigint DEFAULT NULL,
  PRIMARY KEY (`run_uuid`),
  KEY `experiment_id` (`experiment_id`),
  CONSTRAINT `runs_ibfk_1` FOREIGN KEY (`experiment_id`) REFERENCES `experiments` (`experiment_id`),
  CONSTRAINT `runs_chk_1` CHECK ((`status` in (_utf8mb4'SCHEDULED',_utf8mb4'FAILED',_utf8mb4'FINISHED',_utf8mb4'RUNNING',_utf8mb4'KILLED'))),
  CONSTRAINT `runs_lifecycle_stage` CHECK ((`lifecycle_stage` in (_utf8mb4'active',_utf8mb4'deleted'))),
  CONSTRAINT `source_type` CHECK ((`source_type` in (_utf8mb4'NOTEBOOK',_utf8mb4'JOB',_utf8mb4'LOCAL',_utf8mb4'UNKNOWN',_utf8mb4'PROJECT')))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `runs`
--

LOCK TABLES `runs` WRITE;
/*!40000 ALTER TABLE `runs` DISABLE KEYS */;
INSERT INTO `runs` VALUES ('86f3f7c6a2a54692bafb592fd871a6b3','Stroke_Prediction_RandomForest_v1','UNKNOWN','','','root','FINISHED',1751570188449,1751570193347,'','active','file:///mlruns/2/86f3f7c6a2a54692bafb592fd871a6b3/artifacts',2,NULL),('ad3797b88917440ba31793ba47c5fc52','Stroke_Prediction_LogisticRegression_v1','UNKNOWN','','','root','FINISHED',1751570265395,1751570315178,'','active','file:///mlruns/4/ad3797b88917440ba31793ba47c5fc52/artifacts',4,NULL),('b3cf3b20ad7d4285b3a79f668007e67f','Stroke_Prediction_SVM_v1','UNKNOWN','','','root','FINISHED',1751570401391,1751570406599,'','active','file:///mlruns/5/b3cf3b20ad7d4285b3a79f668007e67f/artifacts',5,NULL),('c552b07b144b40a3930a2024878208f7','Stroke_Prediction_LightGBM_v1','UNKNOWN','','','root','FINISHED',1751570132171,1751570140041,'','active','file:///mlruns/1/c552b07b144b40a3930a2024878208f7/artifacts',1,NULL),('e4d6ee7c50cc4dbeb8cf2689608bd94a','Stroke_Prediction_XGBoost_v1','UNKNOWN','','','root','FINISHED',1751570218367,1751570225127,'','active','file:///mlruns/3/e4d6ee7c50cc4dbeb8cf2689608bd94a/artifacts',3,NULL);
/*!40000 ALTER TABLE `runs` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `tags`
--

DROP TABLE IF EXISTS `tags`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(8000) DEFAULT NULL,
  `run_uuid` varchar(32) NOT NULL,
  PRIMARY KEY (`key`,`run_uuid`),
  KEY `index_tags_run_uuid` (`run_uuid`),
  CONSTRAINT `tags_ibfk_1` FOREIGN KEY (`run_uuid`) REFERENCES `runs` (`run_uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `tags`
--

LOCK TABLES `tags` WRITE;
/*!40000 ALTER TABLE `tags` DISABLE KEYS */;
INSERT INTO `tags` VALUES ('mlflow.log-model.history','[{\"run_id\": \"86f3f7c6a2a54692bafb592fd871a6b3\", \"artifact_path\": \"random_forest_model\", \"utc_time_created\": \"2025-07-03 19:16:29.782620\", \"model_uuid\": \"8b1fd7027a3941aaa9fb268498506e1e\", \"flavors\": {\"python_function\": {\"model_path\": \"model.pkl\", \"predict_fn\": \"predict\", \"loader_module\": \"mlflow.sklearn\", \"python_version\": \"3.11.6\", \"env\": {\"conda\": \"conda.yaml\", \"virtualenv\": \"python_env.yaml\"}}, \"sklearn\": {\"pickled_model\": \"model.pkl\", \"sklearn_version\": \"1.6.1\", \"serialization_format\": \"cloudpickle\", \"code\": null}}}]','86f3f7c6a2a54692bafb592fd871a6b3'),('mlflow.log-model.history','[{\"run_id\": \"ad3797b88917440ba31793ba47c5fc52\", \"artifact_path\": \"logistic_regression_model\", \"utc_time_created\": \"2025-07-03 19:18:31.753469\", \"model_uuid\": \"3ef8a4d92db04ae4b94a1bb7f6c9be5a\", \"flavors\": {\"python_function\": {\"model_path\": \"model.pkl\", \"predict_fn\": \"predict\", \"loader_module\": \"mlflow.sklearn\", \"python_version\": \"3.11.6\", \"env\": {\"conda\": \"conda.yaml\", \"virtualenv\": \"python_env.yaml\"}}, \"sklearn\": {\"pickled_model\": \"model.pkl\", \"sklearn_version\": \"1.6.1\", \"serialization_format\": \"cloudpickle\", \"code\": null}}}]','ad3797b88917440ba31793ba47c5fc52'),('mlflow.log-model.history','[{\"run_id\": \"b3cf3b20ad7d4285b3a79f668007e67f\", \"artifact_path\": \"svm_model\", \"utc_time_created\": \"2025-07-03 19:20:02.984715\", \"model_uuid\": \"a4b8a14c5b0f40448699116d85265e62\", \"flavors\": {\"python_function\": {\"model_path\": \"model.pkl\", \"predict_fn\": \"predict\", \"loader_module\": \"mlflow.sklearn\", \"python_version\": \"3.11.6\", \"env\": {\"conda\": \"conda.yaml\", \"virtualenv\": \"python_env.yaml\"}}, \"sklearn\": {\"pickled_model\": \"model.pkl\", \"sklearn_version\": \"1.6.1\", \"serialization_format\": \"cloudpickle\", \"code\": null}}}]','b3cf3b20ad7d4285b3a79f668007e67f'),('mlflow.log-model.history','[{\"run_id\": \"c552b07b144b40a3930a2024878208f7\", \"artifact_path\": \"model\", \"utc_time_created\": \"2025-07-03 19:15:32.791079\", \"model_uuid\": \"71ee08f22f12445392ca564d72b1a876\", \"flavors\": {\"python_function\": {\"loader_module\": \"mlflow.lightgbm\", \"python_version\": \"3.11.6\", \"data\": \"model.pkl\", \"env\": {\"conda\": \"conda.yaml\", \"virtualenv\": \"python_env.yaml\"}}, \"lightgbm\": {\"lgb_version\": \"4.6.0\", \"data\": \"model.pkl\", \"model_class\": \"lightgbm.sklearn.LGBMClassifier\", \"code\": null}}}, {\"run_id\": \"c552b07b144b40a3930a2024878208f7\", \"artifact_path\": \"model\", \"utc_time_created\": \"2025-07-03 19:15:36.626704\", \"model_uuid\": \"ebb02d706af6431f902161a0e604c5ab\", \"flavors\": {\"python_function\": {\"model_path\": \"model.pkl\", \"predict_fn\": \"predict\", \"loader_module\": \"mlflow.sklearn\", \"python_version\": \"3.11.6\", \"env\": {\"conda\": \"conda.yaml\", \"virtualenv\": \"python_env.yaml\"}}, \"sklearn\": {\"pickled_model\": \"model.pkl\", \"sklearn_version\": \"1.6.1\", \"serialization_format\": \"cloudpickle\", \"code\": null}}}]','c552b07b144b40a3930a2024878208f7'),('mlflow.log-model.history','[{\"run_id\": \"e4d6ee7c50cc4dbeb8cf2689608bd94a\", \"artifact_path\": \"model\", \"utc_time_created\": \"2025-07-03 19:16:58.776486\", \"model_uuid\": \"f21c172a6985476099f8f5771e8b2bba\", \"flavors\": {\"python_function\": {\"loader_module\": \"mlflow.xgboost\", \"python_version\": \"3.11.6\", \"data\": \"model.xgb\", \"env\": {\"conda\": \"conda.yaml\", \"virtualenv\": \"python_env.yaml\"}}, \"xgboost\": {\"xgb_version\": \"3.0.2\", \"data\": \"model.xgb\", \"model_class\": \"xgboost.sklearn.XGBClassifier\", \"model_format\": \"xgb\", \"code\": null}}}, {\"run_id\": \"e4d6ee7c50cc4dbeb8cf2689608bd94a\", \"artifact_path\": \"xgboost_model\", \"utc_time_created\": \"2025-07-03 19:17:02.137267\", \"model_uuid\": \"1690ad903410455c8aa30d8caac555e7\", \"flavors\": {\"python_function\": {\"model_path\": \"model.pkl\", \"predict_fn\": \"predict\", \"loader_module\": \"mlflow.sklearn\", \"python_version\": \"3.11.6\", \"env\": {\"conda\": \"conda.yaml\", \"virtualenv\": \"python_env.yaml\"}}, \"sklearn\": {\"pickled_model\": \"model.pkl\", \"sklearn_version\": \"1.6.1\", \"serialization_format\": \"cloudpickle\", \"code\": null}}}]','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('mlflow.runName','Stroke_Prediction_RandomForest_v1','86f3f7c6a2a54692bafb592fd871a6b3'),('mlflow.runName','Stroke_Prediction_LogisticRegression_v1','ad3797b88917440ba31793ba47c5fc52'),('mlflow.runName','Stroke_Prediction_SVM_v1','b3cf3b20ad7d4285b3a79f668007e67f'),('mlflow.runName','Stroke_Prediction_LightGBM_v1','c552b07b144b40a3930a2024878208f7'),('mlflow.runName','Stroke_Prediction_XGBoost_v1','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('mlflow.source.git.commit','625b88275c0ffef94223ef15a54eb83d6ed68c15','86f3f7c6a2a54692bafb592fd871a6b3'),('mlflow.source.git.commit','625b88275c0ffef94223ef15a54eb83d6ed68c15','ad3797b88917440ba31793ba47c5fc52'),('mlflow.source.git.commit','625b88275c0ffef94223ef15a54eb83d6ed68c15','b3cf3b20ad7d4285b3a79f668007e67f'),('mlflow.source.git.commit','625b88275c0ffef94223ef15a54eb83d6ed68c15','c552b07b144b40a3930a2024878208f7'),('mlflow.source.git.commit','625b88275c0ffef94223ef15a54eb83d6ed68c15','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('mlflow.source.name','./work/notebook_experiments/RandomForest_StrokePrediction_MLflow.py','86f3f7c6a2a54692bafb592fd871a6b3'),('mlflow.source.name','./work/notebook_experiments/LogisticRegression_StrokePrediction_MLflow.py','ad3797b88917440ba31793ba47c5fc52'),('mlflow.source.name','./work/notebook_experiments/SVM_StrokePrediction_MLflow.py','b3cf3b20ad7d4285b3a79f668007e67f'),('mlflow.source.name','./work/notebook_experiments/LightGBM_StrokePrediction_MLflow.py','c552b07b144b40a3930a2024878208f7'),('mlflow.source.name','./work/notebook_experiments/XGBoost_StrokePrediction_MLflow.py','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('mlflow.source.type','LOCAL','86f3f7c6a2a54692bafb592fd871a6b3'),('mlflow.source.type','LOCAL','ad3797b88917440ba31793ba47c5fc52'),('mlflow.source.type','LOCAL','b3cf3b20ad7d4285b3a79f668007e67f'),('mlflow.source.type','LOCAL','c552b07b144b40a3930a2024878208f7'),('mlflow.source.type','LOCAL','e4d6ee7c50cc4dbeb8cf2689608bd94a'),('mlflow.user','Thato','86f3f7c6a2a54692bafb592fd871a6b3'),('mlflow.user','Thato','ad3797b88917440ba31793ba47c5fc52'),('mlflow.user','Thato','b3cf3b20ad7d4285b3a79f668007e67f'),('mlflow.user','Thato','c552b07b144b40a3930a2024878208f7'),('mlflow.user','Thato','e4d6ee7c50cc4dbeb8cf2689608bd94a');
/*!40000 ALTER TABLE `tags` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trace_info`
--

DROP TABLE IF EXISTS `trace_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trace_info` (
  `request_id` varchar(50) NOT NULL,
  `experiment_id` int NOT NULL,
  `timestamp_ms` bigint NOT NULL,
  `execution_time_ms` bigint DEFAULT NULL,
  `status` varchar(50) NOT NULL,
  PRIMARY KEY (`request_id`),
  KEY `index_trace_info_experiment_id_timestamp_ms` (`experiment_id`,`timestamp_ms`),
  CONSTRAINT `fk_trace_info_experiment_id` FOREIGN KEY (`experiment_id`) REFERENCES `experiments` (`experiment_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trace_info`
--

LOCK TABLES `trace_info` WRITE;
/*!40000 ALTER TABLE `trace_info` DISABLE KEYS */;
/*!40000 ALTER TABLE `trace_info` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trace_request_metadata`
--

DROP TABLE IF EXISTS `trace_request_metadata`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trace_request_metadata` (
  `key` varchar(250) NOT NULL,
  `value` varchar(8000) DEFAULT NULL,
  `request_id` varchar(50) NOT NULL,
  PRIMARY KEY (`key`,`request_id`),
  KEY `index_trace_request_metadata_request_id` (`request_id`),
  CONSTRAINT `fk_trace_request_metadata_request_id` FOREIGN KEY (`request_id`) REFERENCES `trace_info` (`request_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trace_request_metadata`
--

LOCK TABLES `trace_request_metadata` WRITE;
/*!40000 ALTER TABLE `trace_request_metadata` DISABLE KEYS */;
/*!40000 ALTER TABLE `trace_request_metadata` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `trace_tags`
--

DROP TABLE IF EXISTS `trace_tags`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `trace_tags` (
  `key` varchar(250) NOT NULL,
  `value` varchar(8000) DEFAULT NULL,
  `request_id` varchar(50) NOT NULL,
  PRIMARY KEY (`key`,`request_id`),
  KEY `index_trace_tags_request_id` (`request_id`),
  CONSTRAINT `fk_trace_tags_request_id` FOREIGN KEY (`request_id`) REFERENCES `trace_info` (`request_id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `trace_tags`
--

LOCK TABLES `trace_tags` WRITE;
/*!40000 ALTER TABLE `trace_tags` DISABLE KEYS */;
/*!40000 ALTER TABLE `trace_tags` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping events for database 'mlflow_db'
--

--
-- Dumping routines for database 'mlflow_db'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-07-05  4:38:10
