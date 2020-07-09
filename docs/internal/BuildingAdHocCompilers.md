## Brief notes on how to build a compiler

* log into admin node `ce admin`
* start the builder node: `ce builder start`
* log into builder node: `ce builder login`
* sudo docker run --rm --name gcc.build -v/home/ubuntu/.s3cfg:/root/.s3cfg:ro -ti compilerexplorer/gcc-builder bash build.sh 8.2.0 s3://compiler-explorer/opt/
* Remember to stop the builder node!
  * Log out of the builder node
  * `ce builder stop`
