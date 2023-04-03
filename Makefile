
.PHONY: build
build:
	docker-compose build

.PHONY: runserver
runserver:
	flask --app vectorboss run

.PHONY: shell
shell:
	docker-compose \
	 -f ./docker-compose.yaml \
	 run \
	 --rm \
	 -v ${PWD}:/opt/vectorboss/ \
	 vectorboss \
	 bash -c "/bin/bash --login"

.PHONY: stop
stop:
	docker-compose stop

.PHONY: clean
clean:
	docker-compose rm -fv
