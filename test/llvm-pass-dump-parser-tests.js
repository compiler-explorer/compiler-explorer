import {LlvmPassDumpParser} from '../lib/parsers/llvm-pass-dump-parser';
import * as properties from '../lib/properties';

const languages = {
    'c++': {id: 'c++'},
};

function deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj));
}

describe('llvm-pass-dump-parser filter', function () {
    let llvmPassDumpParser;

    before(() => {
        let fakeProps = new properties.CompilerProps(languages, properties.fakeProps({}));
        let compilerProps = fakeProps.get.bind(fakeProps, 'c++');
        llvmPassDumpParser = new LlvmPassDumpParser(compilerProps);
    });
    // prettier-ignore
    const rawFuncIR = [
        {text: '  # Machine code for function f(S1&, S2 const&): NoPHIs, TracksLiveness, TiedOpsRewritten'},
        {text: 'define dso_local void @f(S1&, S2 const&)(%struct.S1* noundef nonnull align 8 dereferenceable(16) %s1, %struct.S2* noundef nonnull align 8 dereferenceable(16) %s2) #0 !dbg !7 {'},
        {text: 'entry:'},
        {text: '  %s1.addr = alloca %struct.S1*, align 8'},
        {text: '  store %struct.S1* %s1, %struct.S1** %s1.addr, align 8, !tbaa !32'},
        {text: '  call void @llvm.dbg.declare(metadata %struct.S1** %s1.addr, metadata !30, metadata !DIExpression()), !dbg !36'},
        {text: '  call void @llvm.dbg.value(metadata %struct.S1* %s1, metadata !30, metadata !DIExpression()), !dbg !32'},
        {text: '  DBG_VALUE $rdi, $noreg, !"s1", !DIExpression(), debug-location !32; example.cpp:0 line no:7'},
        {text: '  store %struct.S2* %s2, %struct.S2** %s2.addr, align 8, !tbaa !32'},
        {text: '  %0 = load %struct.S2*, %struct.S2** %s2.addr, align 8, !dbg !38, !tbaa !32'},
        {text: '  %a = getelementptr inbounds %struct.S2, %struct.S2* %0, i32 0, i32 0, !dbg !39'},
        {text: '  %1 = load i64, i64* %t, align 8, !dbg !40, !tbaa !41'},
        {text: '  %2 = load %struct.S1*, %struct.S1** %s1.addr, align 8, !dbg !46, !tbaa !32'},
        {text: '  store i64 %1, i64* %t2, align 8, !dbg !49, !tbaa !50'},
        {text: '  %t3 = getelementptr inbounds %struct.Wrapper2, %struct.Wrapper2* %b, i32 0, i32 0, !dbg !54'},
        {text: '  ret void, !dbg !61'},
    ];

    it('should not filter out dbg metadata', function () {
        const options = {filterDebugInfo: false};
        // prettier-ignore
        llvmPassDumpParser
            .applyIrFilters(deepCopy(rawFuncIR), options)
            .should.deep.equal(rawFuncIR);
    });

    it('should filter out dbg metadata too', function () {
        const options = {filterDebugInfo: true};
        // prettier-ignore
        llvmPassDumpParser
            .applyIrFilters(deepCopy(rawFuncIR), options)
            .should.deep.equal([
                {text: '  # Machine code for function f(S1&, S2 const&): NoPHIs, TracksLiveness, TiedOpsRewritten'},
                {text: 'define dso_local void @f(S1&, S2 const&)(%struct.S1* noundef nonnull align 8 dereferenceable(16) %s1, %struct.S2* noundef nonnull align 8 dereferenceable(16) %s2) {'},
                {text: 'entry:'},
                {text: '  %s1.addr = alloca %struct.S1*, align 8'},
                {text: '  store %struct.S1* %s1, %struct.S1** %s1.addr, align 8, !tbaa !32'},
                {text: '  store %struct.S2* %s2, %struct.S2** %s2.addr, align 8, !tbaa !32'},
                {text: '  %0 = load %struct.S2*, %struct.S2** %s2.addr, align 8, !tbaa !32'},
                {text: '  %a = getelementptr inbounds %struct.S2, %struct.S2* %0, i32 0, i32 0'},
                {text: '  %1 = load i64, i64* %t, align 8, !tbaa !41'},
                {text: '  %2 = load %struct.S1*, %struct.S1** %s1.addr, align 8, !tbaa !32'},
                {text: '  store i64 %1, i64* %t2, align 8, !tbaa !50'},
                {text: '  %t3 = getelementptr inbounds %struct.Wrapper2, %struct.Wrapper2* %b, i32 0, i32 0'},
                {text: '  ret void'},
            ]);
    });

    it('should filter out instruction metadata and object attribute group, leave debug instructions in place', function () {
        // 'hide IR metadata' aims to decrease more visual noise than `hide debug info`
        const options = {filterDebugInfo: false, filterIRMetadata: true};
        // prettier-ignore
        llvmPassDumpParser
            .applyIrFilters(deepCopy(rawFuncIR), options)
            .should.deep.equal([
                {text: '  # Machine code for function f(S1&, S2 const&): NoPHIs, TracksLiveness, TiedOpsRewritten'},
                {text: 'define dso_local void @f(S1&, S2 const&)(%struct.S1* noundef nonnull align 8 dereferenceable(16) %s1, %struct.S2* noundef nonnull align 8 dereferenceable(16) %s2) {'},
                {text: 'entry:'},
                {text: '  %s1.addr = alloca %struct.S1*, align 8'},
                {text: '  store %struct.S1* %s1, %struct.S1** %s1.addr, align 8'},
                {text: '  call void @llvm.dbg.declare(metadata %struct.S1** %s1.addr, metadata !30, metadata !DIExpression())'},
                {text: '  call void @llvm.dbg.value(metadata %struct.S1* %s1, metadata !30, metadata !DIExpression())'},
                {text: '  DBG_VALUE $rdi, $noreg, !"s1", !DIExpression(), debug-location !32; example.cpp:0 line no:7'},
                {text: '  store %struct.S2* %s2, %struct.S2** %s2.addr, align 8'},
                {text: '  %0 = load %struct.S2*, %struct.S2** %s2.addr, align 8'},
                {text: '  %a = getelementptr inbounds %struct.S2, %struct.S2* %0, i32 0, i32 0'},
                {text: '  %1 = load i64, i64* %t, align 8'},
                {text: '  %2 = load %struct.S1*, %struct.S1** %s1.addr, align 8'},
                {text: '  store i64 %1, i64* %t2, align 8'},
                {text: '  %t3 = getelementptr inbounds %struct.Wrapper2, %struct.Wrapper2* %b, i32 0, i32 0'},
                {text: '  ret void'},
            ]);
    });
});
