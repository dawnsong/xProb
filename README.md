# xProb

Xiaowei's Machine Learning library for basic probability theory that are working in Databricks notebook with spark backend.

Here are the brief indices of implemented functions for big data:

Tested Env: Databricks 15.4 LTS

1. pcorr4df: fast calculation of Pearson correlation among random variables within a Spark Dataframe, that can have millions of rows
2. bokeh4pcorr: interactive exploring the calculated Pearson correlation coefficient; Ctrl+Click the screenshot to view my shared DataBricks Notebook: [![Interactive covariance matrix exploration using Bokeh, in which the covariance matrix is calculated fastly using Spark.](./images/snapshots/xProb-BokehPearsonCorr__Xiaowei20250323.png)](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/602088969175307/2380861729268519/7374924302554790/latest.html){:target="_blank"}
