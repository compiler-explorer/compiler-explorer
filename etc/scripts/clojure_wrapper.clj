(ns clojure-wrapper
  (:require [clojure.java.io :as io]
            [clojure.pprint :as pp]
            [clojure.string :as str]
            [clojure.walk :as walk])
  (:import [java.io PushbackReader]))

(def help-text
  "Compiler options supported:
   --disable-locals-clearing - Eliminates instructions setting locals to null
   --direct-linking - Eliminates var indirection in fn invocation
   --elide-meta \"[:doc :arglists ...]\" - Drops metadata keys from classfiles
   --omit-macro-meta - Omit metadata from macro-expanded output")

(defn parse-command-line []
  (loop [params {}
         macro-params {}
         positional []
         args *command-line-args*]
    (if (seq args)
      (let [arg (first args)]
        (condp = arg
          "--help"
          (do
            (println help-text)
            (System/exit 0))

          "--macro-expand"
          (recur params (assoc macro-params :macro-expand true)
                 positional (rest args))

          "--omit-macro-meta"
          (recur params (assoc macro-params :print-meta false)
                 positional (rest args))

          "--disable-locals-clearing"
          (recur (assoc params :disable-locals-clearing true)
                 macro-params positional (rest args))

          "--direct-linking"
          (recur (assoc params :direct-linking true)
                 macro-params positional (rest args))

          "--elide-meta"
          (let [elisions (some-> args second read-string)]
            (when-not (and (sequential? elisions)
                           (every? keyword? elisions))
              (println (str "Invalid elide-meta parameter: '" (second args) "'\n")
                       "Must be a string representing a vector of keywords, like \"[:keyword1 :keyword2]\"")
              (System/exit 1))
            (recur (assoc params :elide-meta elisions)
                   macro-params positional (drop 2 args)))

          (if (re-matches #"-.*" arg)
            (do
              (println "Invaliid compiler parameter:" arg)
              (println help-text)
              (System/exit 1))
            (recur params macro-params (conj positional arg) (rest args)))))
      [params macro-params positional])))

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

(defn print-macro-expanson [input-file macro-params]
  (binding [clojure.pprint/*print-pprint-dispatch* clojure.pprint/code-dispatch
            clojure.pprint/*print-right-margin* 60
            clojure.pprint/*print-miser-width* 20
            *print-meta* (:print-meta macro-params true)]
    (doseq [form (forms input-file)]
      (pp/pprint (walk/macroexpand-all form))
      (println))))

(defn compile-input [input-file compiler-options]
  (let [working-dir (path-of-file input-file)
        namespace (read-namespace input-file)
        missing-namespace? (nil? namespace)
        namespace (or namespace "sample")
        compile-filename (io/file working-dir (ns->filename namespace))
        compile-path (path-of-file compile-filename)]
    (.mkdirs (io/file working-dir "classes"))
    (when compile-path
      (.mkdirs (io/file compile-path)))
    (with-open [out (io/writer (io/output-stream compile-filename))]
      (when missing-namespace?
        (let [ns-form (str "(ns " namespace ")")]
          (println "Injecting namespace form on first line:" ns-form)
          (.write out ns-form)))
      (io/copy input-file out))

    (println "Available compiler options: --help --disable-locals-clearing --direct-linking --elide-meta \"[:doc :line]\"")
    (println "Binding *compiler-options* to" compiler-options)
    (binding [*compiler-options* compiler-options]
      (compile (symbol namespace)))))

(let [[compiler-options macro-params positional] (parse-command-line)
      input-file (io/file (first positional))]
  (if (:macro-expand macro-params)
    (print-macro-expanson input-file macro-params)
    (compile-input input-file compiler-options)))
