(ns clojure-wrapper
  (:require [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.io PushbackReader]))

(def help-text
  "Compiler options supported:
   --disable-locals-clearing - Eliminates instructions setting locals to null
   --direct-linking - Eliminates var indirection in fn invocation
   --elide-meta \"[:doc :arglists ...]\" - Drops metadata keys from classfiles ")

(defn parse-command-line []
  (loop [params {}
         positional []
         args *command-line-args*]
    (if (seq args)
      (let [arg (first args)]
        (cond
          (re-matches #"-?-help" arg)
          (do
            (println help-text)
            (System/exit 1))

          (re-matches #"-?-gen-meta" arg)
          (recur (assoc params :gen-meta true)
                 positional (rest args))

          (re-matches #"-?-disable-locals-clearing" arg)
          (recur (assoc params :disable-locals-clearing true)
                 positional (rest args))

          (re-matches #"-?-direct-linking" arg)
          (recur (assoc params :direct-linking true)
                 positional (rest args))

          (re-matches #"-?-elide-meta" arg)
          (let [elisions (some-> args second read-string)]
            (when-not (and (sequential? elisions)
                           (every? keyword? elisions))
              (println (str "Invalid elide-meta parameter: '" (second args) "'\n")
                       "Must be a string representing a vector of keywords, like \"[:keyword1 :keyword2]\"")
              (System/exit 1))
            (recur (assoc params :elide-meta elisions)
                   positional (drop 2 args)))

          (re-matches #"-?-.*" arg)
          (do
            (println "Invaliid compiler parameter:" arg)
            (println help-text)
            (System/exit 1))

          :else
          (recur params (conj positional arg) (rest args))))
      [params positional])))

(defn forms [input-file]
  (with-open [rdr (-> input-file io/reader PushbackReader.)]
    (loop [forms []]
      (if-let [form (try (read rdr) (catch Exception e nil))]
        (recur (conj forms form))
        forms))))

(defn read-namespace [input-file]
  (with-open [rdr (-> input-file io/reader PushbackReader.)]
    (loop []
      (when-let [form (try (read rdr) (catch Exception e nil))]
        (if (and (= 'ns (first form))
                 (symbol? (second form)))
          (-> form second name)
          (recur))))))

(defn ns->filename [namespace]
  (-> namespace
      (str/replace "." "/")
      (str/replace "-" "_")
      (str ".clj")))

(defn path-of-file [file]
  (.getParent file))

(let [[compiler-options positional] (parse-command-line)
      input-file (io/file (first positional))
      working-dir (path-of-file input-file)
      namespace (read-namespace input-file)
      missing-namespace? (nil? namespace)
      namespace (or namespace "sample")
      compile-filename (io/file working-dir (ns->filename namespace))
      compile-path (path-of-file compile-filename)]

  (when (:gen-meta compiler-options)
    (doseq [form (forms input-file)]
      (prn (macroexpand form)))
    (System/exit 0))

  (println "Available compiler options: --help --disable-locals-clearing --direct-linking --elide-meta \"[:doc :line]\"")
  (println "Binding *compiler-options* to" compiler-options)

  (.mkdirs (io/file working-dir "classes"))
  (when compile-path
    (.mkdirs (io/file compile-path)))
  (with-open [out (io/writer (io/output-stream compile-filename))]
    (when missing-namespace?
      (let [ns-form (str "(ns " namespace ")")]
        (println "Injecting namespace form on first line:" ns-form)
        (.write out ns-form)))
    (io/copy input-file out))

  (binding [*compiler-options* compiler-options]
    (compile (symbol namespace))))

