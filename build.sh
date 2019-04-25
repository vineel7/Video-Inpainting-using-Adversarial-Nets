#!/bin/bash -e

echo "MAVEN_OPTS='-Xmx1024m'" > ~/.mavenrc

if [ $TESSA2_API_KEY ]; then
  echo "TESSA2_API_KEY found in environment variable"
  mvn clean package tessa:update
else
  mvn clean package
fi
