export const UrlTestCases = [
    {
        remoteUrl: 'https://example.com/path/to/resource',
        language: 'cpp',
        expectedId: 'example_com_path_to_resource_cpp',
    },
    {
        remoteUrl: 'https://example.com',
        language: 'java',
        expectedId: 'example_com__java',
    },
    {
        remoteUrl: 'https://sub.domain.com/a/b/c/',
        language: 'rust',
        expectedId: 'sub_domain_com_a_b_c__rust',
    },
    {
        remoteUrl: 'https://godbolt.org:443/gpu',
        language: 'c++',
        expectedId: 'godbolt_org_gpu_c++',
    },
    {
        remoteUrl: 'https://godbolt.org:443/winprod',
        language: 'c++',
        expectedId: 'godbolt_org_winprod_c++',
    },
    {
        remoteUrl: 'https://godbolt.org:443',
        language: 'c++',
        expectedId: 'godbolt_org__c++',
    },
];
