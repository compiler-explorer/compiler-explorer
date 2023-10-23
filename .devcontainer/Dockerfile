ARG NODE_MAJOR_VERSION

FROM mcr.microsoft.com/devcontainers/javascript-node:${NODE_MAJOR_VERSION}

ENV EDITOR="code -w" VISUAL="code -w"

# install system dependencies for cypress (https://docs.cypress.io/guides/continuous-integration/introduction)
RUN apt update && apt install -y libgtk2.0-0 libgtk-3-0 libgbm-dev libnotify-dev libnss3 libxss1 libasound2 libxtst6 xauth xvfb

# uncomment to install additional npm packages
# RUN su node -c 'npm i -g cowsay@1.5.0'
