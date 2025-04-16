/// <reference path="./node_modules/vite/client.d.ts" />

declare module "*.pug" {
    declare const fun: (args?: Record<PropertyKey, any>) => string;
    export default fun;
}
