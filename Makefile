SERVICE_NAME=xdmviewer
# $sha is provided by jenkins
BUILDER_TAG?=$(or $(sha),$(SERVICE_NAME)-builder)
IMAGE_TAG=$(SERVICE_NAME)-img
BUILD_DOCKERFILE_NAME=Dockerfile.build.mt

default: ci

login:
ifeq ($(ARTIFACTORY_USER), )
	@echo "make sure that ARTIFACTORY_USER is set appropriately in your environment"
	exit 1
endif
ifeq ($(ARTIFACTORY_API_TOKEN), )
	@echo "make sure that ARTIFACTORY_API_TOKEN is set appropriately in your environment"
	exit 1
endif
	@echo docker login -u ARTIFACTORY_USER -p ARTIFACTORY_API_TOKEN docker-asr-release.dr.corp.adobe.com
	@docker login -u $(ARTIFACTORY_USER) -p $(ARTIFACTORY_API_TOKEN) docker-asr-release.dr.corp.adobe.com

# The goal is to build the deployable and run any available tests.
# The ci stage executes all the steps defined in build and then launches the service image.
ci: build
	docker run --rm $(IMAGE_TAG) bash -c 'which java && java -version'
	echo "Success"

# The goal is run the service with remote debugging enabled on port 4000
debug: build
	docker run -it -e ENVIRONMENT_NAME=local -e REGION_NAME=local -e JVM_OPTIONS="-Xdebug -Xrunjdwp:transport=dt_socket,server=y,address=4000" -p 8080:8080 -p 4000:4000 $(IMAGE_TAG)

pre-deploy-build:
	echo "Nothing is defined in pre-deploy-build step"

# Not actually used in Porter builds, but simulates what Porter does. For local development purposes.
# The goal is to build the deployable and run any available tests.
# The ci-like-porter stage executes all the steps defined in build-like-porter and then launches the service image.
ci-like-porter: build-like-porter
	docker run --rm $(IMAGE_TAG) bash -c 'which java && java -version'
	echo "Success"

# Not actually used in Porter builds, but simulates what Porter does. For local development purposes.
build-like-porter: login
	docker build -t $(BUILDER_TAG) --build-arg ARTIFACTORY_USER --build-arg ARTIFACTORY_API_TOKEN -f $(BUILD_DOCKERFILE_NAME) .
	docker run $(BUILDER_TAG) | docker build --tag $(IMAGE_TAG) -

build: login
	# A separate image for build allows the process to avoid dependencies with the build machine.
	docker build -t $(BUILDER_TAG) -f $(BUILD_DOCKERFILE_NAME) .
	# Runs the image generated in the above step to create the actual deployable artifact (i.e. jar file).
	# docker run -t -v `pwd`:/build $(BUILDER_TAG)
	# In the above build step, maven starts of by downloading all dependencies and
	# then proceeds to build the artifact. This can be time consuming. To reuse mvn cache
	# please use the following command instead. This will ensure, a docker volume is
	# responsible for incrementally saving the dependencies and avoids the time taken to download
	# all dependencies everytime.
	# Note: There can be issues when applications include snapshot repositories. Try removing the docker volume
	# and rebuilding. cmd: docker volume rm m2
	# Since the build happens on shared build machines, the m2 volume will be shared between builds from different projects.
	# In case there is a need to have dedicated mvn repo cache for the project use a project specific name for the volume.
	# for example m2_asrexample instead of m2
	@docker run -e TESSA2_API_KEY=$(TESSA2_API_KEY) -e ARTIFACTORY_USER -e ARTIFACTORY_API_TOKEN -v m2:/root/.m2 -v `pwd`:/build $(BUILDER_TAG)
	# Builds the docker image for running the service.
	docker build -t $(IMAGE_TAG) .
	echo "Success"

post-deploy-build:
	echo "Nothing is defined in post-deploy-build step"

clean-room:
	# Docker Clean Room setup - https://wiki.corp.adobe.com/x/khu5TQ
	sh clean-room.sh
