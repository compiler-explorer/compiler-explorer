(ns clojure-wrapper
  (:require [clojure.java.io :as io]
            [clojure.pprint :as pp]
            [clojure.string :as str]
            [clojure.walk :as walk])
  (:import [java.io PushbackReader]))

(def help-text
  "Compiler options supported:
   -h   --help - Shows this text and flags sent to compiler
   -dl  --direct-linking - Eliminates var indirection in fn invocation
   -dlc --disable-locals-clearing - Eliminates instructions setting locals to null
   -em  --elide-meta [:doc,:arglists,:added,:file,...] - Drops metadata keys from classfiles
   -omm --omit-macro-meta - Omit metadata from macro-expanded output")

(defn parse-command-line []
  (loop [params {}
         positional []
         ignored []
         args *command-line-args*]
    (if-let [arg (first args)]
      (case arg
        ("-h" "--help")
        (recur (assoc params :show-help true)
               positional ignored (rest args))

        "--macro-expand"
        (recur (assoc params :macro-expand true)
               positional ignored (rest args))

        ("-omm" "--omit-macro-meta")
        (recur (assoc params :print-meta false)
               positional ignored (rest args))

        ("-dlc" "--disable-locals-clearing")
        (recur (assoc params :disable-locals-clearing true)
               positional ignored (rest args))

        ("-dl" "--direct-linking")
        (recur (assoc params :direct-linking true)
               positional ignored (rest args))

        ("-em" "--elide-meta")
        (let [elisions (try (some-> args second read-string) (catch Exception _e))]
          (when-not (and (sequential? elisions)
                         (every? keyword? elisions))
            (println (str "Invalid elide-meta parameter: '" (second args) "'\n")
                     "-em flag must be followed by a vector of keywords, like '-em [:doc,:arglists]'")
            (System/exit 1))
          (recur (assoc params :elide-meta elisions)
                 positional ignored (drop 2 args)))

        (if (or (re-matches #"-.*" arg)
                (not (re-matches #".*\.clj" arg)))
          (recur params positional (conj ignored arg) (rest args))
          (recur params (conj positional arg) ignored (rest args))))
      [params positional ignored])))

(defn forms
  ([input-file]
   ;; Default is to load all forms while file is open
   (forms input-file doall))
  ([input-file extract]
   (with-open [rdr (-> input-file io/reader PushbackReader.)]
     (->> #(try (read rdr) (catch Exception _e nil))
          (repeatedly)
          (take-while some?)
          extract))))

(defn read-namespace [input-file]
  (let [parse-ns-name (fn [forms]
                        (some->> forms
                                 (filter (fn [form]
                                           (and (= 'ns (first form))
                                                (symbol? (second form)))))
                                 first   ;; ns form
                                 second  ;; namespace symbol
                                 name))]
    (forms input-file parse-ns-name)))

(defn ns->filename [namespace]
  (-> namespace
      (str/replace "." "/")
      (str/replace "-" "_")
      (str ".clj")))

(defn path-of-file [file]
  (.getParent file))

(defn print-macro-expansion [input-file macro-params]
  (binding [clojure.pprint/*print-pprint-dispatch* clojure.pprint/code-dispatch
            clojure.pprint/*print-right-margin* 60
            clojure.pprint/*print-miser-width* 20
            *print-meta* (:print-meta macro-params true)]
    (doseq [form (forms input-file)]
      (pp/pprint (walk/macroexpand-all form))
      (println))))

(defn compile-input [input-file {:keys [show-help] :as params}]
  (let [working-dir (path-of-file input-file)
        namespace (read-namespace input-file)
        missing-namespace? (nil? namespace)
        namespace (or namespace "sample")
        compile-filename (io/file working-dir (ns->filename namespace))
        compile-path (path-of-file compile-filename)
        compiler-options (select-keys params
                                      [:disable-locals-clearing
                                       :direct-linking
                                       :elide-meta])]
    (.mkdirs (io/file working-dir "classes"))
    (when compile-path
      (.mkdirs (io/file compile-path)))
    (with-open [out (io/writer (io/output-stream compile-filename))]
      (when missing-namespace?
        (let [ns-form (str "(ns " namespace ")")]
          (println "Injecting namespace form on first line:" ns-form)
          (.write out ns-form)))
      (io/copy input-file out))

    (when show-help
      (when (seq *compiler-options*)
        (println "*compiler-options* set via environment:" *compiler-options*))
      (when (seq compiler-options)
        (println "*compiler-options* set via flags:" compiler-options)))
    (binding [*compiler-options* (merge *compiler-options* compiler-options)]
      (compile (symbol namespace)))))

(let [[params positional ignored] (parse-command-line)
      input-file (io/file (first positional))]
  (if (:macro-expand params)
    (print-macro-expansion input-file params)
    (let [count-ignored (count ignored)]
      (doseq [param ignored]
        (println (format "unrecognized option '%s' ignored" param)))
      (when (pos-int? count-ignored)
        (println (format "%d warning%s found" count-ignored
                         (if (= 1 count-ignored) "" "s"))))
      (when (or (:show-help params)
                (pos-int? count-ignored))
        (println help-text))
      (compile-input input-file params))))
