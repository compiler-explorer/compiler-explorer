// Copyright (c) 2026, Compiler Explorer Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

package ce.nativeimage;

import java.io.IOException;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.graalvm.nativeimage.hosted.Feature;

import com.oracle.svm.core.util.InterruptImageBuilding;
import com.oracle.svm.hosted.FeatureImpl.AfterCompilationAccessImpl;
import com.oracle.svm.hosted.FeatureImpl.BeforeAnalysisAccessImpl;
import com.oracle.svm.hosted.code.HostedImageHeapConstantPatch;

import jdk.graal.compiler.code.CompilationResult;
import jdk.graal.compiler.util.json.JsonWriter;
import jdk.vm.ci.code.site.Call;
import jdk.vm.ci.code.site.DataPatch;
import jdk.vm.ci.meta.ResolvedJavaMethod;

public final class ExplorerFeature implements Feature {
    private final Set<String> classes = new HashSet<>();

    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        var internal = (BeforeAnalysisAccessImpl) access;
        var directory = Path.of(System.getProperty("ce.nativeimage.classes"));
        try (var files = Files.walk(directory)) {
            for (var file : files.filter(p -> p.toString().endsWith(".class")).sorted().toList()) {
                var relative = directory.relativize(file).toString();
                var name = relative.substring(0, relative.length() - 6).replace('/', '.');
                if (name.equals("module-info") || name.endsWith("package-info")) continue;
                classes.add(name);
                var type = access.findClassByName(name);
                if (type == null) throw new IllegalArgumentException("Cannot load " + name);
                if (!type.isInterface() && !Modifier.isAbstract(type.getModifiers())) {
                    internal.registerAsUnsafeAllocated(type);
                }
                for (var method : type.getDeclaredMethods()) {
                    if (!Modifier.isAbstract(method.getModifiers()) && !Modifier.isNative(method.getModifiers())) {
                        internal.registerAsRoot(method, true, "Compiler Explorer method");
                    }
                }
                for (var constructor : type.getDeclaredConstructors()) {
                    internal.registerAsRoot(constructor, true, "Compiler Explorer constructor");
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void afterCompilation(AfterCompilationAccess access) {
        var methods = new ArrayList<Object>();
        var tasks = ((AfterCompilationAccessImpl) access).getCompilationTasks().stream()
                .sorted(Comparator.comparing(task -> task.method.format("%H.%n(%p)%r"))).toList();
        for (var task : tasks) {
            var method = task.method;
            if (!classes.contains(method.getDeclaringClass().toJavaName())) continue;
            CompilationResult result = task.result;
            var patches = new ArrayList<Object>();
            for (Object rawPoint : (List<?>) result.getInfopoints()) {
                if (rawPoint instanceof Call call) {
                    var target = call.target instanceof ResolvedJavaMethod m
                            ? m.format("%H.%n(%p)%r") : call.target.toString();
                    patches.add(Map.of("offset", call.pcOffset, "kind", "call", "target", target, "direct", call.direct));
                }
            }
            for (Object rawPatch : (List<?>) result.getDataPatches()) {
                DataPatch patch = (DataPatch) rawPatch;
                patches.add(Map.of("offset", patch.pcOffset, "kind", "data", "target", patch.reference.toString()));
            }
            for (var annotation : result.getCodeAnnotations()) {
                if (annotation instanceof HostedImageHeapConstantPatch patch) {
                    patches.add(Map.of("offset", patch.getPosition(), "kind", "data", "target", "image heap constant"));
                }
            }
            var mappings = new ArrayList<Object>();
            for (var mapping : result.getSourceMappings()) {
                var position = mapping.getSourcePosition();
                // Only map positions in submitted classes; library inlining may refer to unrelated source files.
                while (position != null && !classes.contains(position.getMethod().getDeclaringClass().toJavaName())) {
                    position = position.getCaller();
                }
                if (position == null) continue;
                var table = position.getMethod().getLineNumberTable();
                if (table == null) continue;
                int line = table.getLineNumber(position.getBCI());
                if (line > 0) {
                    mappings.add(Map.of("start", mapping.getStartOffset(), "end", mapping.getEndOffset(), "line", line));
                }
            }
            methods.add(Map.of("name", method.format("%H.%n(%p)%r"),
                    "code", HexFormat.of().formatHex(Arrays.copyOf(result.getTargetCode(), result.getTargetCodeSize())),
                    "patches", patches, "mappings", mappings));
        }
        var output = Path.of(System.getProperty("ce.nativeimage.output"));
        try (var writer = new JsonWriter(output)) {
            writer.print(Map.of("version", 1, "architecture", System.getProperty("os.arch"), "methods", methods));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        throw new InterruptImageBuilding("Compiler Explorer extraction complete");
    }
}
