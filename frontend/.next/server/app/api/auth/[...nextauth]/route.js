"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(() => {
var exports = {};
exports.id = "app/api/auth/[...nextauth]/route";
exports.ids = ["app/api/auth/[...nextauth]/route"];
exports.modules = {

/***/ "mongodb":
/*!**************************!*\
  !*** external "mongodb" ***!
  \**************************/
/***/ ((module) => {

module.exports = require("mongodb");

/***/ }),

/***/ "../../client/components/action-async-storage.external":
/*!*******************************************************************************!*\
  !*** external "next/dist/client/components/action-async-storage.external.js" ***!
  \*******************************************************************************/
/***/ ((module) => {

module.exports = require("next/dist/client/components/action-async-storage.external.js");

/***/ }),

/***/ "../../client/components/request-async-storage.external":
/*!********************************************************************************!*\
  !*** external "next/dist/client/components/request-async-storage.external.js" ***!
  \********************************************************************************/
/***/ ((module) => {

module.exports = require("next/dist/client/components/request-async-storage.external.js");

/***/ }),

/***/ "../../client/components/static-generation-async-storage.external":
/*!******************************************************************************************!*\
  !*** external "next/dist/client/components/static-generation-async-storage.external.js" ***!
  \******************************************************************************************/
/***/ ((module) => {

module.exports = require("next/dist/client/components/static-generation-async-storage.external.js");

/***/ }),

/***/ "next/dist/compiled/next-server/app-page.runtime.dev.js":
/*!*************************************************************************!*\
  !*** external "next/dist/compiled/next-server/app-page.runtime.dev.js" ***!
  \*************************************************************************/
/***/ ((module) => {

module.exports = require("next/dist/compiled/next-server/app-page.runtime.dev.js");

/***/ }),

/***/ "next/dist/compiled/next-server/app-route.runtime.dev.js":
/*!**************************************************************************!*\
  !*** external "next/dist/compiled/next-server/app-route.runtime.dev.js" ***!
  \**************************************************************************/
/***/ ((module) => {

module.exports = require("next/dist/compiled/next-server/app-route.runtime.dev.js");

/***/ }),

/***/ "assert":
/*!*************************!*\
  !*** external "assert" ***!
  \*************************/
/***/ ((module) => {

module.exports = require("assert");

/***/ }),

/***/ "buffer":
/*!*************************!*\
  !*** external "buffer" ***!
  \*************************/
/***/ ((module) => {

module.exports = require("buffer");

/***/ }),

/***/ "crypto":
/*!*************************!*\
  !*** external "crypto" ***!
  \*************************/
/***/ ((module) => {

module.exports = require("crypto");

/***/ }),

/***/ "events":
/*!*************************!*\
  !*** external "events" ***!
  \*************************/
/***/ ((module) => {

module.exports = require("events");

/***/ }),

/***/ "http":
/*!***********************!*\
  !*** external "http" ***!
  \***********************/
/***/ ((module) => {

module.exports = require("http");

/***/ }),

/***/ "https":
/*!************************!*\
  !*** external "https" ***!
  \************************/
/***/ ((module) => {

module.exports = require("https");

/***/ }),

/***/ "querystring":
/*!******************************!*\
  !*** external "querystring" ***!
  \******************************/
/***/ ((module) => {

module.exports = require("querystring");

/***/ }),

/***/ "url":
/*!**********************!*\
  !*** external "url" ***!
  \**********************/
/***/ ((module) => {

module.exports = require("url");

/***/ }),

/***/ "util":
/*!***********************!*\
  !*** external "util" ***!
  \***********************/
/***/ ((module) => {

module.exports = require("util");

/***/ }),

/***/ "zlib":
/*!***********************!*\
  !*** external "zlib" ***!
  \***********************/
/***/ ((module) => {

module.exports = require("zlib");

/***/ }),

/***/ "(rsc)/./node_modules/next/dist/build/webpack/loaders/next-app-loader.js?name=app%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute&page=%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute&appPaths=&pagePath=private-next-app-dir%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute.ts&appDir=D%3A%5Cfyp%5Csem1_finalized_malaika%5Csem1%5Cfrontend%5Capp&pageExtensions=tsx&pageExtensions=ts&pageExtensions=jsx&pageExtensions=js&rootDir=D%3A%5Cfyp%5Csem1_finalized_malaika%5Csem1%5Cfrontend&isDev=true&tsconfigPath=tsconfig.json&basePath=&assetPrefix=&nextConfigOutput=&preferredRegion=&middlewareConfig=e30%3D!":
/*!**************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************!*\
  !*** ./node_modules/next/dist/build/webpack/loaders/next-app-loader.js?name=app%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute&page=%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute&appPaths=&pagePath=private-next-app-dir%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute.ts&appDir=D%3A%5Cfyp%5Csem1_finalized_malaika%5Csem1%5Cfrontend%5Capp&pageExtensions=tsx&pageExtensions=ts&pageExtensions=jsx&pageExtensions=js&rootDir=D%3A%5Cfyp%5Csem1_finalized_malaika%5Csem1%5Cfrontend&isDev=true&tsconfigPath=tsconfig.json&basePath=&assetPrefix=&nextConfigOutput=&preferredRegion=&middlewareConfig=e30%3D! ***!
  \**************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   originalPathname: () => (/* binding */ originalPathname),\n/* harmony export */   patchFetch: () => (/* binding */ patchFetch),\n/* harmony export */   requestAsyncStorage: () => (/* binding */ requestAsyncStorage),\n/* harmony export */   routeModule: () => (/* binding */ routeModule),\n/* harmony export */   serverHooks: () => (/* binding */ serverHooks),\n/* harmony export */   staticGenerationAsyncStorage: () => (/* binding */ staticGenerationAsyncStorage)\n/* harmony export */ });\n/* harmony import */ var next_dist_server_future_route_modules_app_route_module_compiled__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! next/dist/server/future/route-modules/app-route/module.compiled */ \"(rsc)/./node_modules/next/dist/server/future/route-modules/app-route/module.compiled.js\");\n/* harmony import */ var next_dist_server_future_route_modules_app_route_module_compiled__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(next_dist_server_future_route_modules_app_route_module_compiled__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var next_dist_server_future_route_kind__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/dist/server/future/route-kind */ \"(rsc)/./node_modules/next/dist/server/future/route-kind.js\");\n/* harmony import */ var next_dist_server_lib_patch_fetch__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/dist/server/lib/patch-fetch */ \"(rsc)/./node_modules/next/dist/server/lib/patch-fetch.js\");\n/* harmony import */ var next_dist_server_lib_patch_fetch__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_dist_server_lib_patch_fetch__WEBPACK_IMPORTED_MODULE_2__);\n/* harmony import */ var D_fyp_sem1_finalized_malaika_sem1_frontend_app_api_auth_nextauth_route_ts__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./app/api/auth/[...nextauth]/route.ts */ \"(rsc)/./app/api/auth/[...nextauth]/route.ts\");\n\n\n\n\n// We inject the nextConfigOutput here so that we can use them in the route\n// module.\nconst nextConfigOutput = \"\"\nconst routeModule = new next_dist_server_future_route_modules_app_route_module_compiled__WEBPACK_IMPORTED_MODULE_0__.AppRouteRouteModule({\n    definition: {\n        kind: next_dist_server_future_route_kind__WEBPACK_IMPORTED_MODULE_1__.RouteKind.APP_ROUTE,\n        page: \"/api/auth/[...nextauth]/route\",\n        pathname: \"/api/auth/[...nextauth]\",\n        filename: \"route\",\n        bundlePath: \"app/api/auth/[...nextauth]/route\"\n    },\n    resolvedPagePath: \"D:\\\\fyp\\\\sem1_finalized_malaika\\\\sem1\\\\frontend\\\\app\\\\api\\\\auth\\\\[...nextauth]\\\\route.ts\",\n    nextConfigOutput,\n    userland: D_fyp_sem1_finalized_malaika_sem1_frontend_app_api_auth_nextauth_route_ts__WEBPACK_IMPORTED_MODULE_3__\n});\n// Pull out the exports that we need to expose from the module. This should\n// be eliminated when we've moved the other routes to the new format. These\n// are used to hook into the route.\nconst { requestAsyncStorage, staticGenerationAsyncStorage, serverHooks } = routeModule;\nconst originalPathname = \"/api/auth/[...nextauth]/route\";\nfunction patchFetch() {\n    return (0,next_dist_server_lib_patch_fetch__WEBPACK_IMPORTED_MODULE_2__.patchFetch)({\n        serverHooks,\n        staticGenerationAsyncStorage\n    });\n}\n\n\n//# sourceMappingURL=app-route.js.map//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKHJzYykvLi9ub2RlX21vZHVsZXMvbmV4dC9kaXN0L2J1aWxkL3dlYnBhY2svbG9hZGVycy9uZXh0LWFwcC1sb2FkZXIuanM/bmFtZT1hcHAlMkZhcGklMkZhdXRoJTJGJTVCLi4ubmV4dGF1dGglNUQlMkZyb3V0ZSZwYWdlPSUyRmFwaSUyRmF1dGglMkYlNUIuLi5uZXh0YXV0aCU1RCUyRnJvdXRlJmFwcFBhdGhzPSZwYWdlUGF0aD1wcml2YXRlLW5leHQtYXBwLWRpciUyRmFwaSUyRmF1dGglMkYlNUIuLi5uZXh0YXV0aCU1RCUyRnJvdXRlLnRzJmFwcERpcj1EJTNBJTVDZnlwJTVDc2VtMV9maW5hbGl6ZWRfbWFsYWlrYSU1Q3NlbTElNUNmcm9udGVuZCU1Q2FwcCZwYWdlRXh0ZW5zaW9ucz10c3gmcGFnZUV4dGVuc2lvbnM9dHMmcGFnZUV4dGVuc2lvbnM9anN4JnBhZ2VFeHRlbnNpb25zPWpzJnJvb3REaXI9RCUzQSU1Q2Z5cCU1Q3NlbTFfZmluYWxpemVkX21hbGFpa2ElNUNzZW0xJTVDZnJvbnRlbmQmaXNEZXY9dHJ1ZSZ0c2NvbmZpZ1BhdGg9dHNjb25maWcuanNvbiZiYXNlUGF0aD0mYXNzZXRQcmVmaXg9Jm5leHRDb25maWdPdXRwdXQ9JnByZWZlcnJlZFJlZ2lvbj0mbWlkZGxld2FyZUNvbmZpZz1lMzAlM0QhIiwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7OztBQUFzRztBQUN2QztBQUNjO0FBQ3dDO0FBQ3JIO0FBQ0E7QUFDQTtBQUNBLHdCQUF3QixnSEFBbUI7QUFDM0M7QUFDQSxjQUFjLHlFQUFTO0FBQ3ZCO0FBQ0E7QUFDQTtBQUNBO0FBQ0EsS0FBSztBQUNMO0FBQ0E7QUFDQSxZQUFZO0FBQ1osQ0FBQztBQUNEO0FBQ0E7QUFDQTtBQUNBLFFBQVEsaUVBQWlFO0FBQ3pFO0FBQ0E7QUFDQSxXQUFXLDRFQUFXO0FBQ3RCO0FBQ0E7QUFDQSxLQUFLO0FBQ0w7QUFDdUg7O0FBRXZIIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vbXktdjAtcHJvamVjdC8/YWY2YyJdLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBBcHBSb3V0ZVJvdXRlTW9kdWxlIH0gZnJvbSBcIm5leHQvZGlzdC9zZXJ2ZXIvZnV0dXJlL3JvdXRlLW1vZHVsZXMvYXBwLXJvdXRlL21vZHVsZS5jb21waWxlZFwiO1xuaW1wb3J0IHsgUm91dGVLaW5kIH0gZnJvbSBcIm5leHQvZGlzdC9zZXJ2ZXIvZnV0dXJlL3JvdXRlLWtpbmRcIjtcbmltcG9ydCB7IHBhdGNoRmV0Y2ggYXMgX3BhdGNoRmV0Y2ggfSBmcm9tIFwibmV4dC9kaXN0L3NlcnZlci9saWIvcGF0Y2gtZmV0Y2hcIjtcbmltcG9ydCAqIGFzIHVzZXJsYW5kIGZyb20gXCJEOlxcXFxmeXBcXFxcc2VtMV9maW5hbGl6ZWRfbWFsYWlrYVxcXFxzZW0xXFxcXGZyb250ZW5kXFxcXGFwcFxcXFxhcGlcXFxcYXV0aFxcXFxbLi4ubmV4dGF1dGhdXFxcXHJvdXRlLnRzXCI7XG4vLyBXZSBpbmplY3QgdGhlIG5leHRDb25maWdPdXRwdXQgaGVyZSBzbyB0aGF0IHdlIGNhbiB1c2UgdGhlbSBpbiB0aGUgcm91dGVcbi8vIG1vZHVsZS5cbmNvbnN0IG5leHRDb25maWdPdXRwdXQgPSBcIlwiXG5jb25zdCByb3V0ZU1vZHVsZSA9IG5ldyBBcHBSb3V0ZVJvdXRlTW9kdWxlKHtcbiAgICBkZWZpbml0aW9uOiB7XG4gICAgICAgIGtpbmQ6IFJvdXRlS2luZC5BUFBfUk9VVEUsXG4gICAgICAgIHBhZ2U6IFwiL2FwaS9hdXRoL1suLi5uZXh0YXV0aF0vcm91dGVcIixcbiAgICAgICAgcGF0aG5hbWU6IFwiL2FwaS9hdXRoL1suLi5uZXh0YXV0aF1cIixcbiAgICAgICAgZmlsZW5hbWU6IFwicm91dGVcIixcbiAgICAgICAgYnVuZGxlUGF0aDogXCJhcHAvYXBpL2F1dGgvWy4uLm5leHRhdXRoXS9yb3V0ZVwiXG4gICAgfSxcbiAgICByZXNvbHZlZFBhZ2VQYXRoOiBcIkQ6XFxcXGZ5cFxcXFxzZW0xX2ZpbmFsaXplZF9tYWxhaWthXFxcXHNlbTFcXFxcZnJvbnRlbmRcXFxcYXBwXFxcXGFwaVxcXFxhdXRoXFxcXFsuLi5uZXh0YXV0aF1cXFxccm91dGUudHNcIixcbiAgICBuZXh0Q29uZmlnT3V0cHV0LFxuICAgIHVzZXJsYW5kXG59KTtcbi8vIFB1bGwgb3V0IHRoZSBleHBvcnRzIHRoYXQgd2UgbmVlZCB0byBleHBvc2UgZnJvbSB0aGUgbW9kdWxlLiBUaGlzIHNob3VsZFxuLy8gYmUgZWxpbWluYXRlZCB3aGVuIHdlJ3ZlIG1vdmVkIHRoZSBvdGhlciByb3V0ZXMgdG8gdGhlIG5ldyBmb3JtYXQuIFRoZXNlXG4vLyBhcmUgdXNlZCB0byBob29rIGludG8gdGhlIHJvdXRlLlxuY29uc3QgeyByZXF1ZXN0QXN5bmNTdG9yYWdlLCBzdGF0aWNHZW5lcmF0aW9uQXN5bmNTdG9yYWdlLCBzZXJ2ZXJIb29rcyB9ID0gcm91dGVNb2R1bGU7XG5jb25zdCBvcmlnaW5hbFBhdGhuYW1lID0gXCIvYXBpL2F1dGgvWy4uLm5leHRhdXRoXS9yb3V0ZVwiO1xuZnVuY3Rpb24gcGF0Y2hGZXRjaCgpIHtcbiAgICByZXR1cm4gX3BhdGNoRmV0Y2goe1xuICAgICAgICBzZXJ2ZXJIb29rcyxcbiAgICAgICAgc3RhdGljR2VuZXJhdGlvbkFzeW5jU3RvcmFnZVxuICAgIH0pO1xufVxuZXhwb3J0IHsgcm91dGVNb2R1bGUsIHJlcXVlc3RBc3luY1N0b3JhZ2UsIHN0YXRpY0dlbmVyYXRpb25Bc3luY1N0b3JhZ2UsIHNlcnZlckhvb2tzLCBvcmlnaW5hbFBhdGhuYW1lLCBwYXRjaEZldGNoLCAgfTtcblxuLy8jIHNvdXJjZU1hcHBpbmdVUkw9YXBwLXJvdXRlLmpzLm1hcCJdLCJuYW1lcyI6W10sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///(rsc)/./node_modules/next/dist/build/webpack/loaders/next-app-loader.js?name=app%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute&page=%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute&appPaths=&pagePath=private-next-app-dir%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute.ts&appDir=D%3A%5Cfyp%5Csem1_finalized_malaika%5Csem1%5Cfrontend%5Capp&pageExtensions=tsx&pageExtensions=ts&pageExtensions=jsx&pageExtensions=js&rootDir=D%3A%5Cfyp%5Csem1_finalized_malaika%5Csem1%5Cfrontend&isDev=true&tsconfigPath=tsconfig.json&basePath=&assetPrefix=&nextConfigOutput=&preferredRegion=&middlewareConfig=e30%3D!\n");

/***/ }),

/***/ "(rsc)/./app/api/auth/[...nextauth]/route.ts":
/*!*********************************************!*\
  !*** ./app/api/auth/[...nextauth]/route.ts ***!
  \*********************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   GET: () => (/* binding */ handler),\n/* harmony export */   POST: () => (/* binding */ handler)\n/* harmony export */ });\n/* harmony import */ var next_auth__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! next-auth */ \"(rsc)/./node_modules/next-auth/index.js\");\n/* harmony import */ var next_auth__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(next_auth__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var _lib_auth_config__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @/lib/auth-config */ \"(rsc)/./lib/auth-config.ts\");\n\n\nconst handler = next_auth__WEBPACK_IMPORTED_MODULE_0___default()(_lib_auth_config__WEBPACK_IMPORTED_MODULE_1__.authOptions);\n\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKHJzYykvLi9hcHAvYXBpL2F1dGgvWy4uLm5leHRhdXRoXS9yb3V0ZS50cyIsIm1hcHBpbmdzIjoiOzs7Ozs7OztBQUFnQztBQUNlO0FBRS9DLE1BQU1FLFVBQVVGLGdEQUFRQSxDQUFDQyx5REFBV0E7QUFFTSIsInNvdXJjZXMiOlsid2VicGFjazovL215LXYwLXByb2plY3QvLi9hcHAvYXBpL2F1dGgvWy4uLm5leHRhdXRoXS9yb3V0ZS50cz9jOGE0Il0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBOZXh0QXV0aCBmcm9tIFwibmV4dC1hdXRoXCJcclxuaW1wb3J0IHsgYXV0aE9wdGlvbnMgfSBmcm9tIFwiQC9saWIvYXV0aC1jb25maWdcIlxyXG5cclxuY29uc3QgaGFuZGxlciA9IE5leHRBdXRoKGF1dGhPcHRpb25zKVxyXG5cclxuZXhwb3J0IHsgaGFuZGxlciBhcyBHRVQsIGhhbmRsZXIgYXMgUE9TVCB9XHJcbiJdLCJuYW1lcyI6WyJOZXh0QXV0aCIsImF1dGhPcHRpb25zIiwiaGFuZGxlciIsIkdFVCIsIlBPU1QiXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///(rsc)/./app/api/auth/[...nextauth]/route.ts\n");

/***/ }),

/***/ "(rsc)/./lib/auth-config.ts":
/*!****************************!*\
  !*** ./lib/auth-config.ts ***!
  \****************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   authOptions: () => (/* binding */ authOptions)\n/* harmony export */ });\n/* harmony import */ var next_auth_providers_google__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! next-auth/providers/google */ \"(rsc)/./node_modules/next-auth/providers/google.js\");\n/* harmony import */ var next_auth_providers_credentials__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next-auth/providers/credentials */ \"(rsc)/./node_modules/next-auth/providers/credentials.js\");\n/* harmony import */ var mongodb__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! mongodb */ \"mongodb\");\n/* harmony import */ var mongodb__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(mongodb__WEBPACK_IMPORTED_MODULE_2__);\n/* harmony import */ var bcryptjs__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! bcryptjs */ \"(rsc)/./node_modules/bcryptjs/index.js\");\n\n\n\n\nconst client = new mongodb__WEBPACK_IMPORTED_MODULE_2__.MongoClient(process.env.MONGO_URI);\nconst db = client.db();\nconst authOptions = {\n    providers: [\n        // Google login\n        (0,next_auth_providers_google__WEBPACK_IMPORTED_MODULE_0__[\"default\"])({\n            clientId: process.env.GOOGLE_CLIENT_ID || \"\",\n            clientSecret: process.env.GOOGLE_CLIENT_SECRET || \"\"\n        }),\n        // Email / password login\n        (0,next_auth_providers_credentials__WEBPACK_IMPORTED_MODULE_1__[\"default\"])({\n            name: \"Credentials\",\n            credentials: {\n                email: {\n                    label: \"Email\",\n                    type: \"text\"\n                },\n                password: {\n                    label: \"Password\",\n                    type: \"password\"\n                }\n            },\n            async authorize (credentials) {\n                try {\n                    // Validate email format\n                    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;\n                    if (!emailRegex.test(credentials.email)) return null;\n                    await client.connect();\n                    const users = db.collection(\"users\");\n                    const user = await users.findOne({\n                        email: credentials.email\n                    });\n                    if (!user || !user.password_hash) return null;\n                    const valid = await bcryptjs__WEBPACK_IMPORTED_MODULE_3__[\"default\"].compare(credentials.password, user.password_hash);\n                    if (!valid) return null;\n                    return {\n                        id: user.user_id,\n                        email: user.email,\n                        name: user.username,\n                        role: user.role || \"user\"\n                    };\n                } catch (error) {\n                    console.error(\"Auth error:\", error);\n                    return null;\n                }\n            }\n        })\n    ],\n    pages: {\n        signIn: \"/signin\",\n        error: \"/signin\"\n    },\n    callbacks: {\n        // Add user to DB on first Google login\n        async signIn ({ user, account }) {\n            if (account?.provider === \"google\") {\n                await client.connect();\n                const users = db.collection(\"users\");\n                const existing = await users.findOne({\n                    email: user.email\n                });\n                if (!existing) {\n                    await users.insertOne({\n                        user_id: crypto.randomUUID(),\n                        email: user.email,\n                        username: user.name,\n                        password_hash: \"\",\n                        role: \"user\",\n                        profile_data: {},\n                        is_active: true,\n                        created_at: new Date(),\n                        updated_at: new Date()\n                    });\n                }\n            }\n            return true;\n        },\n        async jwt ({ token, user, account }) {\n            // When user first signs in, get their database user_id\n            if (user) {\n                try {\n                    await client.connect();\n                    const users = db.collection(\"users\");\n                    const dbUser = await users.findOne({\n                        $or: [\n                            {\n                                email: user.email\n                            },\n                            {\n                                user_id: user.id\n                            } // For credentials login\n                        ]\n                    });\n                    if (dbUser) {\n                        // Store database user_id in token (not Google ID)\n                        token.userId = dbUser.user_id;\n                        token.role = dbUser.role || user.role || \"user\";\n                    } else if (user.id) {\n                        // Fallback to user.id if database lookup fails (shouldn't happen)\n                        token.userId = user.id;\n                        token.role = user.role || \"user\";\n                    }\n                } catch (error) {\n                    console.error(\"Error in JWT callback:\", error);\n                    // Fallback to user.id if database lookup fails\n                    if (user.id) {\n                        token.userId = user.id;\n                        token.role = user.role || \"user\";\n                    }\n                }\n            }\n            // On token refresh, preserve existing userId and role\n            // (user is undefined on refresh, so we keep the existing token values)\n            return token;\n        },\n        async session ({ session, token }) {\n            // Use userId from token (database user_id) instead of token.sub (which might be Google ID)\n            if (token?.userId) {\n                session.user.id = token.userId;\n            } else if (token?.sub) {\n                // Fallback to token.sub if userId not available\n                session.user.id = token.sub;\n            }\n            if (token?.role) {\n                session.user.role = token.role;\n            }\n            return session;\n        }\n    },\n    session: {\n        strategy: \"jwt\"\n    },\n    secret: process.env.NEXTAUTH_SECRET || process.env.AUTH_SECRET\n};\n// Validate that secret is set\nif (!process.env.NEXTAUTH_SECRET && !process.env.AUTH_SECRET) {\n    console.error(\"⚠️ WARNING: NEXTAUTH_SECRET is not set in environment variables!\");\n    console.error(\"Please add NEXTAUTH_SECRET to your .env.local file\");\n    console.error(\"Current working directory:\", process.cwd());\n    console.error(\"NODE_ENV:\", \"development\");\n} else {\n    console.log(\"✅ NEXTAUTH_SECRET is loaded successfully\");\n}\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,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LElBQUk7WUFDekM7WUFDQSxPQUFPc0I7UUFDVDtJQUNGO0lBRUFBLFNBQVM7UUFBRUUsVUFBVTtJQUFNO0lBQzNCQyxRQUFRdkQsUUFBUUMsR0FBRyxDQUFDdUQsZUFBZSxJQUFJeEQsUUFBUUMsR0FBRyxDQUFDd0QsV0FBVztBQUNoRSxFQUFDO0FBRUQsOEJBQThCO0FBQzlCLElBQUksQ0FBQ3pELFFBQVFDLEdBQUcsQ0FBQ3VELGVBQWUsSUFBSSxDQUFDeEQsUUFBUUMsR0FBRyxDQUFDd0QsV0FBVyxFQUFFO0lBQzVEekIsUUFBUUQsS0FBSyxDQUFDO0lBQ2RDLFFBQVFELEtBQUssQ0FBQztJQUNkQyxRQUFRRCxLQUFLLENBQUMsOEJBQThCL0IsUUFBUTBELEdBQUc7SUFDdkQxQixRQUFRRCxLQUFLLENBQUMsYUE5SWhCO0FBK0lBLE9BQU87SUFDTEMsUUFBUTJCLEdBQUcsQ0FBQztBQUNkIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8vbXktdjAtcHJvamVjdC8uL2xpYi9hdXRoLWNvbmZpZy50cz9lZmUzIl0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB0eXBlIHsgTmV4dEF1dGhPcHRpb25zIH0gZnJvbSBcIm5leHQtYXV0aFwiXHJcbmltcG9ydCBHb29nbGVQcm92aWRlciBmcm9tIFwibmV4dC1hdXRoL3Byb3ZpZGVycy9nb29nbGVcIlxyXG5pbXBvcnQgQ3JlZGVudGlhbHNQcm92aWRlciBmcm9tIFwibmV4dC1hdXRoL3Byb3ZpZGVycy9jcmVkZW50aWFsc1wiXHJcbmltcG9ydCB7IE1vbmdvQ2xpZW50IH0gZnJvbSBcIm1vbmdvZGJcIlxyXG5pbXBvcnQgYmNyeXB0IGZyb20gXCJiY3J5cHRqc1wiXHJcblxyXG5jb25zdCBjbGllbnQgPSBuZXcgTW9uZ29DbGllbnQocHJvY2Vzcy5lbnYuTU9OR09fVVJJISlcclxuY29uc3QgZGIgPSBjbGllbnQuZGIoKVxyXG5cclxuZXhwb3J0IGNvbnN0IGF1dGhPcHRpb25zOiBOZXh0QXV0aE9wdGlvbnMgPSB7XHJcbiAgcHJvdmlkZXJzOiBbXHJcbiAgICAvLyBHb29nbGUgbG9naW5cclxuICAgIEdvb2dsZVByb3ZpZGVyKHtcclxuICAgICAgY2xpZW50SWQ6IHByb2Nlc3MuZW52LkdPT0dMRV9DTElFTlRfSUQgfHwgXCJcIixcclxuICAgICAgY2xpZW50U2VjcmV0OiBwcm9jZXNzLmVudi5HT09HTEVfQ0xJRU5UX1NFQ1JFVCB8fCBcIlwiLFxyXG4gICAgfSksXHJcblxyXG4gICAgLy8gRW1haWwgLyBwYXNzd29yZCBsb2dpblxyXG4gICAgQ3JlZGVudGlhbHNQcm92aWRlcih7XHJcbiAgICAgIG5hbWU6IFwiQ3JlZGVudGlhbHNcIixcclxuICAgICAgY3JlZGVudGlhbHM6IHtcclxuICAgICAgICBlbWFpbDogeyBsYWJlbDogXCJFbWFpbFwiLCB0eXBlOiBcInRleHRcIiB9LFxyXG4gICAgICAgIHBhc3N3b3JkOiB7IGxhYmVsOiBcIlBhc3N3b3JkXCIsIHR5cGU6IFwicGFzc3dvcmRcIiB9LFxyXG4gICAgICB9LFxyXG4gICAgICBhc3luYyBhdXRob3JpemUoY3JlZGVudGlhbHMpIHtcclxuICAgICAgICB0cnkge1xyXG4gICAgICAgICAgLy8gVmFsaWRhdGUgZW1haWwgZm9ybWF0XHJcbiAgICAgICAgICBjb25zdCBlbWFpbFJlZ2V4ID0gL15bXlxcc0BdK0BbXlxcc0BdK1xcLlteXFxzQF0rJC9cclxuICAgICAgICAgIGlmICghZW1haWxSZWdleC50ZXN0KGNyZWRlbnRpYWxzIS5lbWFpbCkpIHJldHVybiBudWxsXHJcblxyXG4gICAgICAgICAgYXdhaXQgY2xpZW50LmNvbm5lY3QoKVxyXG4gICAgICAgICAgY29uc3QgdXNlcnMgPSBkYi5jb2xsZWN0aW9uKFwidXNlcnNcIilcclxuICAgICAgICAgIGNvbnN0IHVzZXIgPSBhd2FpdCB1c2Vycy5maW5kT25lKHsgZW1haWw6IGNyZWRlbnRpYWxzIS5lbWFpbCB9KVxyXG5cclxuICAgICAgICAgIGlmICghdXNlciB8fCAhdXNlci5wYXNzd29yZF9oYXNoKSByZXR1cm4gbnVsbFxyXG4gICAgICAgICAgY29uc3QgdmFsaWQgPSBhd2FpdCBiY3J5cHQuY29tcGFyZShjcmVkZW50aWFscyEucGFzc3dvcmQsIHVzZXIucGFzc3dvcmRfaGFzaClcclxuICAgICAgICAgIGlmICghdmFsaWQpIHJldHVybiBudWxsXHJcblxyXG4gICAgICAgICAgcmV0dXJuIHtcclxuICAgICAgICAgICAgaWQ6IHVzZXIudXNlcl9pZCxcclxuICAgICAgICAgICAgZW1haWw6IHVzZXIuZW1haWwsXHJcbiAgICAgICAgICAgIG5hbWU6IHVzZXIudXNlcm5hbWUsXHJcbiAgICAgICAgICAgIHJvbGU6IHVzZXIucm9sZSB8fCBcInVzZXJcIixcclxuICAgICAgICAgIH1cclxuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xyXG4gICAgICAgICAgY29uc29sZS5lcnJvcihcIkF1dGggZXJyb3I6XCIsIGVycm9yKVxyXG4gICAgICAgICAgcmV0dXJuIG51bGxcclxuICAgICAgICB9XHJcbiAgICAgIH0sXHJcbiAgICB9KSxcclxuICBdLFxyXG5cclxuICBwYWdlczoge1xyXG4gICAgc2lnbkluOiBcIi9zaWduaW5cIixcclxuICAgIGVycm9yOiBcIi9zaWduaW5cIixcclxuICB9LFxyXG5cclxuICBjYWxsYmFja3M6IHtcclxuICAgIC8vIEFkZCB1c2VyIHRvIERCIG9uIGZpcnN0IEdvb2dsZSBsb2dpblxyXG4gICAgYXN5bmMgc2lnbkluKHsgdXNlciwgYWNjb3VudCB9KSB7XHJcbiAgICAgIGlmIChhY2NvdW50Py5wcm92aWRlciA9PT0gXCJnb29nbGVcIikge1xyXG4gICAgICAgIGF3YWl0IGNsaWVudC5jb25uZWN0KClcclxuICAgICAgICBjb25zdCB1c2VycyA9IGRiLmNvbGxlY3Rpb24oXCJ1c2Vyc1wiKVxyXG4gICAgICAgIGNvbnN0IGV4aXN0aW5nID0gYXdhaXQgdXNlcnMuZmluZE9uZSh7IGVtYWlsOiB1c2VyLmVtYWlsIH0pXHJcblxyXG4gICAgICAgIGlmICghZXhpc3RpbmcpIHtcclxuICAgICAgICAgIGF3YWl0IHVzZXJzLmluc2VydE9uZSh7XHJcbiAgICAgICAgICAgIHVzZXJfaWQ6IGNyeXB0by5yYW5kb21VVUlEKCksXHJcbiAgICAgICAgICAgIGVtYWlsOiB1c2VyLmVtYWlsLFxyXG4gICAgICAgICAgICB1c2VybmFtZTogdXNlci5uYW1lLFxyXG4gICAgICAgICAgICBwYXNzd29yZF9oYXNoOiBcIlwiLFxyXG4gICAgICAgICAgICByb2xlOiBcInVzZXJcIixcclxuICAgICAgICAgICAgcHJvZmlsZV9kYXRhOiB7fSxcclxuICAgICAgICAgICAgaXNfYWN0aXZlOiB0cnVlLFxyXG4gICAgICAgICAgICBjcmVhdGVkX2F0OiBuZXcgRGF0ZSgpLFxyXG4gICAgICAgICAgICB1cGRhdGVkX2F0OiBuZXcgRGF0ZSgpLFxyXG4gICAgICAgICAgfSlcclxuICAgICAgICB9XHJcbiAgICAgIH1cclxuICAgICAgcmV0dXJuIHRydWVcclxuICAgIH0sXHJcblxyXG4gICAgYXN5bmMgand0KHsgdG9rZW4sIHVzZXIsIGFjY291bnQgfSkge1xyXG4gICAgICAvLyBXaGVuIHVzZXIgZmlyc3Qgc2lnbnMgaW4sIGdldCB0aGVpciBkYXRhYmFzZSB1c2VyX2lkXHJcbiAgICAgIGlmICh1c2VyKSB7XHJcbiAgICAgICAgdHJ5IHtcclxuICAgICAgICAgIGF3YWl0IGNsaWVudC5jb25uZWN0KClcclxuICAgICAgICAgIGNvbnN0IHVzZXJzID0gZGIuY29sbGVjdGlvbihcInVzZXJzXCIpXHJcbiAgICAgICAgICBjb25zdCBkYlVzZXIgPSBhd2FpdCB1c2Vycy5maW5kT25lKHsgXHJcbiAgICAgICAgICAgICRvcjogW1xyXG4gICAgICAgICAgICAgIHsgZW1haWw6IHVzZXIuZW1haWwgfSxcclxuICAgICAgICAgICAgICB7IHVzZXJfaWQ6ICh1c2VyIGFzIGFueSkuaWQgfSAvLyBGb3IgY3JlZGVudGlhbHMgbG9naW5cclxuICAgICAgICAgICAgXVxyXG4gICAgICAgICAgfSlcclxuICAgICAgICAgIFxyXG4gICAgICAgICAgaWYgKGRiVXNlcikge1xyXG4gICAgICAgICAgICAvLyBTdG9yZSBkYXRhYmFzZSB1c2VyX2lkIGluIHRva2VuIChub3QgR29vZ2xlIElEKVxyXG4gICAgICAgICAgICB0b2tlbi51c2VySWQgPSBkYlVzZXIudXNlcl9pZFxyXG4gICAgICAgICAgICB0b2tlbi5yb2xlID0gZGJVc2VyLnJvbGUgfHwgKHVzZXIgYXMgYW55KS5yb2xlIHx8IFwidXNlclwiXHJcbiAgICAgICAgICB9IGVsc2UgaWYgKCh1c2VyIGFzIGFueSkuaWQpIHtcclxuICAgICAgICAgICAgLy8gRmFsbGJhY2sgdG8gdXNlci5pZCBpZiBkYXRhYmFzZSBsb29rdXAgZmFpbHMgKHNob3VsZG4ndCBoYXBwZW4pXHJcbiAgICAgICAgICAgIHRva2VuLnVzZXJJZCA9ICh1c2VyIGFzIGFueSkuaWRcclxuICAgICAgICAgICAgdG9rZW4ucm9sZSA9ICh1c2VyIGFzIGFueSkucm9sZSB8fCBcInVzZXJcIlxyXG4gICAgICAgICAgfVxyXG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XHJcbiAgICAgICAgICBjb25zb2xlLmVycm9yKFwiRXJyb3IgaW4gSldUIGNhbGxiYWNrOlwiLCBlcnJvcilcclxuICAgICAgICAgIC8vIEZhbGxiYWNrIHRvIHVzZXIuaWQgaWYgZGF0YWJhc2UgbG9va3VwIGZhaWxzXHJcbiAgICAgICAgICBpZiAoKHVzZXIgYXMgYW55KS5pZCkge1xyXG4gICAgICAgICAgICB0b2tlbi51c2VySWQgPSAodXNlciBhcyBhbnkpLmlkXHJcbiAgICAgICAgICAgIHRva2VuLnJvbGUgPSAodXNlciBhcyBhbnkpLnJvbGUgfHwgXCJ1c2VyXCJcclxuICAgICAgICAgIH1cclxuICAgICAgICB9XHJcbiAgICAgIH1cclxuICAgICAgLy8gT24gdG9rZW4gcmVmcmVzaCwgcHJlc2VydmUgZXhpc3RpbmcgdXNlcklkIGFuZCByb2xlXHJcbiAgICAgIC8vICh1c2VyIGlzIHVuZGVmaW5lZCBvbiByZWZyZXNoLCBzbyB3ZSBrZWVwIHRoZSBleGlzdGluZyB0b2tlbiB2YWx1ZXMpXHJcbiAgICAgIHJldHVybiB0b2tlblxyXG4gICAgfSxcclxuXHJcbiAgICBhc3luYyBzZXNzaW9uKHsgc2Vzc2lvbiwgdG9rZW4gfSkge1xyXG4gICAgICAvLyBVc2UgdXNlcklkIGZyb20gdG9rZW4gKGRhdGFiYXNlIHVzZXJfaWQpIGluc3RlYWQgb2YgdG9rZW4uc3ViICh3aGljaCBtaWdodCBiZSBHb29nbGUgSUQpXHJcbiAgICAgIGlmICh0b2tlbj8udXNlcklkKSB7XHJcbiAgICAgICAgKHNlc3Npb24udXNlciBhcyBhbnkpLmlkID0gdG9rZW4udXNlcklkXHJcbiAgICAgIH0gZWxzZSBpZiAodG9rZW4/LnN1Yikge1xyXG4gICAgICAgIC8vIEZhbGxiYWNrIHRvIHRva2VuLnN1YiBpZiB1c2VySWQgbm90IGF2YWlsYWJsZVxyXG4gICAgICAgIChzZXNzaW9uLnVzZXIgYXMgYW55KS5pZCA9IHRva2VuLnN1YlxyXG4gICAgICB9XHJcbiAgICAgIGlmICh0b2tlbj8ucm9sZSkge1xyXG4gICAgICAgIChzZXNzaW9uLnVzZXIgYXMgYW55KS5yb2xlID0gdG9rZW4ucm9sZVxyXG4gICAgICB9XHJcbiAgICAgIHJldHVybiBzZXNzaW9uXHJcbiAgICB9LFxyXG4gIH0sXHJcblxyXG4gIHNlc3Npb246IHsgc3RyYXRlZ3k6IFwiand0XCIgfSxcclxuICBzZWNyZXQ6IHByb2Nlc3MuZW52Lk5FWFRBVVRIX1NFQ1JFVCB8fCBwcm9jZXNzLmVudi5BVVRIX1NFQ1JFVCxcclxufVxyXG5cclxuLy8gVmFsaWRhdGUgdGhhdCBzZWNyZXQgaXMgc2V0XHJcbmlmICghcHJvY2Vzcy5lbnYuTkVYVEFVVEhfU0VDUkVUICYmICFwcm9jZXNzLmVudi5BVVRIX1NFQ1JFVCkge1xyXG4gIGNvbnNvbGUuZXJyb3IoXCLimqDvuI8gV0FSTklORzogTkVYVEFVVEhfU0VDUkVUIGlzIG5vdCBzZXQgaW4gZW52aXJvbm1lbnQgdmFyaWFibGVzIVwiKVxyXG4gIGNvbnNvbGUuZXJyb3IoXCJQbGVhc2UgYWRkIE5FWFRBVVRIX1NFQ1JFVCB0byB5b3VyIC5lbnYubG9jYWwgZmlsZVwiKVxyXG4gIGNvbnNvbGUuZXJyb3IoXCJDdXJyZW50IHdvcmtpbmcgZGlyZWN0b3J5OlwiLCBwcm9jZXNzLmN3ZCgpKVxyXG4gIGNvbnNvbGUuZXJyb3IoXCJOT0RFX0VOVjpcIiwgcHJvY2Vzcy5lbnYuTk9ERV9FTlYpXHJcbn0gZWxzZSB7XHJcbiAgY29uc29sZS5sb2coXCLinIUgTkVYVEFVVEhfU0VDUkVUIGlzIGxvYWRlZCBzdWNjZXNzZnVsbHlcIilcclxufVxyXG4iXSwibmFtZXMiOlsiR29vZ2xlUHJvdmlkZXIiLCJDcmVkZW50aWFsc1Byb3ZpZGVyIiwiTW9uZ29DbGllbnQiLCJiY3J5cHQiLCJjbGllbnQiLCJwcm9jZXNzIiwiZW52IiwiTU9OR09fVVJJIiwiZGIiLCJhdXRoT3B0aW9ucyIsInByb3ZpZGVycyIsImNsaWVudElkIiwiR09PR0xFX0NMSUVOVF9JRCIsImNsaWVudFNlY3JldCIsIkdPT0dMRV9DTElFTlRfU0VDUkVUIiwibmFtZSIsImNyZWRlbnRpYWxzIiwiZW1haWwiLCJsYWJlbCIsInR5cGUiLCJwYXNzd29yZCIsImF1dGhvcml6ZSIsImVtYWlsUmVnZXgiLCJ0ZXN0IiwiY29ubmVjdCIsInVzZXJzIiwiY29sbGVjdGlvbiIsInVzZXIiLCJmaW5kT25lIiwicGFzc3dvcmRfaGFzaCIsInZhbGlkIiwiY29tcGFyZSIsImlkIiwidXNlcl9pZCIsInVzZXJuYW1lIiwicm9sZSIsImVycm9yIiwiY29uc29sZSIsInBhZ2VzIiwic2lnbkluIiwiY2FsbGJhY2tzIiwiYWNjb3VudCIsInByb3ZpZGVyIiwiZXhpc3RpbmciLCJpbnNlcnRPbmUiLCJjcnlwdG8iLCJyYW5kb21VVUlEIiwicHJvZmlsZV9kYXRhIiwiaXNfYWN0aXZlIiwiY3JlYXRlZF9hdCIsIkRhdGUiLCJ1cGRhdGVkX2F0Iiwiand0IiwidG9rZW4iLCJkYlVzZXIiLCIkb3IiLCJ1c2VySWQiLCJzZXNzaW9uIiwic3ViIiwic3RyYXRlZ3kiLCJzZWNyZXQiLCJORVhUQVVUSF9TRUNSRVQiLCJBVVRIX1NFQ1JFVCIsImN3ZCIsImxvZyJdLCJzb3VyY2VSb290IjoiIn0=\n//# sourceURL=webpack-internal:///(rsc)/./lib/auth-config.ts\n");

/***/ })

};
;

// load runtime
var __webpack_require__ = require("../../../../webpack-runtime.js");
__webpack_require__.C(exports);
var __webpack_exec__ = (moduleId) => (__webpack_require__(__webpack_require__.s = moduleId))
var __webpack_exports__ = __webpack_require__.X(0, ["vendor-chunks/next","vendor-chunks/next-auth","vendor-chunks/@babel","vendor-chunks/jose","vendor-chunks/openid-client","vendor-chunks/oauth","vendor-chunks/@panva","vendor-chunks/yallist","vendor-chunks/preact-render-to-string","vendor-chunks/bcryptjs","vendor-chunks/preact","vendor-chunks/oidc-token-hash","vendor-chunks/object-hash","vendor-chunks/lru-cache","vendor-chunks/cookie"], () => (__webpack_exec__("(rsc)/./node_modules/next/dist/build/webpack/loaders/next-app-loader.js?name=app%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute&page=%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute&appPaths=&pagePath=private-next-app-dir%2Fapi%2Fauth%2F%5B...nextauth%5D%2Froute.ts&appDir=D%3A%5Cfyp%5Csem1_finalized_malaika%5Csem1%5Cfrontend%5Capp&pageExtensions=tsx&pageExtensions=ts&pageExtensions=jsx&pageExtensions=js&rootDir=D%3A%5Cfyp%5Csem1_finalized_malaika%5Csem1%5Cfrontend&isDev=true&tsconfigPath=tsconfig.json&basePath=&assetPrefix=&nextConfigOutput=&preferredRegion=&middlewareConfig=e30%3D!")));
module.exports = __webpack_exports__;

})();