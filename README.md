# xdmviewer

[![orca-service](https://img.shields.io/badge/orca-service-blue.svg?style=flat)](https://orca.ethos.corp.adobe.com/services)
[![moonbeam](https://img.shields.io/badge/ethos-moonbeam-yellow.svg?style=flat)](https://moonbeam.ethos.corp.adobe.com/experience-platform/xdm-viewer)

This is an ASR java application that listens on port 8080, with the apis detailed below. It uses [docker-java](https://git.corp.adobe.com/ASR/docker-java) as base docker image.

This project requires Java 8, and Maven 3.5 or greater.

### Build Container

Building the container is a multi-step process. To learn more about this refer to the following [wiki page](https://wiki.corp.adobe.com/display/CTDxE/make+build+target).

**Note**: The Dockerfile will attempt to use the artifact `target/xdmviewer-*.jar`. If your target jar is named differently or if this regex may be ambigious, please make the appropriate changes in your Dockerfile.

##### Local development

Please refer to the [Local Development](https://wiki.corp.adobe.com/display/CTDxE/DxE+-+Anonymous+access+removal+in+Artifactory#DxE-AnonymousaccessremovalinArtifactory-LocalDevelopment) section of the Artifactory authentication wiki for instructions on setting the ARTIFACTORY_USER and ARTIFACTORY_API_TOKEN environment variables before running the below commands. Your generated service is already configured for Artifactory authentication and needs no changes, but the remainder of that wiki contains more details on how the authentication works in Ethos.

##### Mac users

```
Note: set ARTIFACTORY_USER and ARTIFACTORY_API_TOKEN before running below command
make build
```

##### Windows users

```
Please refer to the [Local Development](https://wiki.corp.adobe.com/display/CTDxE/DxE+-+Anonymous+access+removal+in+Artifactory#DxE-AnonymousaccessremovalinArtifactory-LocalDevelopment) section of the Artifactory authentication wiki for instructions on setting the ARTIFACTORY_USER and ARTIFACTORY_API_TOKEN environment variables before running the below commands. Your generated service is already configured for Artifactory authentication and needs no changes, but the remainder of that wiki contains more details on how the authentication works in Ethos.
docker login -u $(ARTIFACTORY_USER) -p $(ARTIFACTORY_API_TOKEN) docker-asr-release.dr.corp.adobe.com
docker build -t xdmviewer-builder -f Dockerfile.build.mt .
docker run -v m2:/root/.m2 -v <absolute_path_to_source_code>:/build -e ARTIFACTORY_USER -e ARTIFACTORY_API_TOKEN xdmviewer-builder
docker build -t xdmviewer-img .
```

### Run Container

##### Using docker compose

Docker compose is a convenient way of running docker containers. For launching the application using docker-compose, ensure the builds steps are already executed.

```
docker-compose up --build

Remote debugging using docker compose is enabled. https://wiki.corp.adobe.com/display/ethos/Bootcamp+Content+Container+-+101#BootcampContentContainer-101-RemoteDebugging
helps you how to use the feature.
```

##### Using docker command

```
docker run -it -e ENVIRONMENT_NAME=<dev|cd|qa|sqa|stage|prod|local> \
               -e REGION_NAME=<ap-south-1|ap-southeast-1|ap-southeast-2|ap-northeast-1|ap-northeast-2|eu-central-1|eu-west-1|sa-east-1|us-east-1|us-west-1|us-west-2|local> \
               -p 8080:8080 -p 8082:8082 xdmviewer-img
```

To run container locally use:

```
docker run -it -e ENVIRONMENT_NAME=local -e REGION_NAME=local -p 8080:8080 -p 8082:8082 xdmviewer-img
```

To run container with newrelic and jvm params use:

```
docker run -it -e ENVIRONMENT_NAME=local -e REGION_NAME=local \
               -e REPLACE_NEWRELIC_APP=xdmviewer -e REPLACE_NEWRELIC_LICENSE=<newrelic_license_key> \
               -e JVM_OPTIONS="-Xmx2048m -Xms256m" \
               -p 8080:8080 -p 8082:8082 xdmviewer-img
```

To debug container using remote debugging.

```
make debug
Use https://wiki.corp.adobe.com/display/ethos/Bootcamp+Content+Container+-+101#BootcampContentContainer-101-RemoteDebugging for setting up remote debugging.
```

To view container logs, run following command:

```
docker logs <container id>
```

Docker clean room setup:

To ensure that we're starting fresh (useful when you're doing a training session and/or trying to debug a local set up), it's best that we start with a 'clean room' and purge any local images and volumes that could introduce any potential 'contaminants' in our setup. You can read more on the following [wiki](https://wiki.corp.adobe.com/x/khu5TQ). Here is the command for docker clean room setup:

```
make clean-room
```

### List of available APIs

API | Description
--- | ---
`GET /xdmviewer/myfirstapi` | Returns a json payload with a message.
`GET /xdmviewer/version` | A default API added by ASR. Returns a json structure with version information.
`GET /xdmviewer/ping` | A default API added by ASR. Returns string 'pong'. Used for basic healthcheck.
`GET /version` | A default API added by ASR. Returns a json structure with version information.
`GET /ping` | A default API added by ASR. Returns string 'pong'. Used for basic healthcheck.


APIs can be accessed via curl command: `curl http://localhost:8080/<API>`




### Swagger Docs 
 Go to &lt;host_url&gt;/swagger-ui.html to browse and explore the apis.

### Environment Variables


Several environment variables can be used to configure ASR java application.

Refer to the following [wiki](https://wiki.corp.adobe.com/display/CTDxE/docker-java) for more details.

### Tessa 2.0

To enable TESSA, you may need to [setup](https://git.corp.adobe.com/SharedCloud/tessa-maven-plugin#plugin-execution) `TESSA2_API_KEY` as OrCA `build_env_vars` in your service_spec [file](https://git.corp.adobe.com/adobe-platform/service-spec/blob/45dec163fd4b0d8694714dcd675d37d524b9a67a/spec.yaml#L140).

### References

  * Base image: https://git.corp.adobe.com/ASR/bbc-factory/blob/master/README.md