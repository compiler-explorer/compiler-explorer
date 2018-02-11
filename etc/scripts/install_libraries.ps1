
Add-Type -assembly "system.io.compression.filesystem"

$opt = "c:/opt"
$tmp = "C:/tmp"

function prepare {
    if (![System.IO.Directory]::Exists($tmp)) {
        [System.IO.Directory]::CreateDirectory($tmp)
    }

    $dest = "$opt/libs";
    if (![System.IO.Directory]::Exists($dest)) {
        [System.IO.Directory]::CreateDirectory($dest)
    }
}

function fetch {
    Param ([string] $url)

    $dest = getDownloadFilepath -url $url

    [System.Console]::WriteLine("Downloading $url")
    Invoke-WebRequest -Uri $url -OutFile $dest
}

function getDownloadFilepath {
    Param ([string] $url)

    $filename = [System.IO.Path]::GetFileName($url)
    return "$tmp/$filename"
}

function fetchAndUnzip {
    Param ([string] $url, [string] $dest)

    fetch -url $url

    $file = getDownloadFilepath -url $url

    [System.Console]::WriteLine("Extracting $file to $dest")
    [io.compression.zipfile]::ExtractToDirectory($file, $dest)
}

function fetchAndUnzipStripOne {
    Param ([string] $url, [string] $dest)
    
    fetchAndUnzip -url $url -dest $dest

    $folders = Get-ChildItem "$dest"
    
    foreach ($folder in $folders) {
        if (!($folder -eq ".") -And !($folder -eq "..")) {
            $contents = Get-ChildItem "$dest/$folder"

            foreach ($item in $contents) {
                if (!($item -eq ".") -And !($item -eq "..")) {
                    [System.IO.Directory]::Move("$dest/$folder/$item", "$dest/$item")
                }
            }

            [System.IO.Directory]::Delete("$dest/$folder")
        }
    }
}

function install_boost {
    Param ([string] $version)

    $versionUnderscored = $version.replace(".", "_")
    $dest = "$opt/libs"

    if (![System.IO.Directory]::Exists("$dest/boost_$versionUnderscored")) {
        $url = "https://dl.bintray.com/boostorg/release/$version/source/boost_$versionUnderscored.zip"
        fetchAndUnzip -url $url -dest $dest
    }
}

function get_or_sync {
    Param ([string] $dir, [string] $url)

    if (![System.IO.Directory]::Exists("$opt/$dir")) {
        git clone "$url" "$opt/$dir"
    } else {
	    git -C "$opt/$dir" pull
    }
}

function get_if_not_there {
    Param ([string] $dir, [string] $url)

    if (![System.IO.Directory]::Exists("$opt/$dir")) {
        [System.IO.Directory]::CreateDirectory("$opt/$dir")

        fetchAndUnzipStripOne -url $url -dest "$opt/$dir"
    }
}

function get_github_versioned_and_trunk {
    Param ([string] $dir, [string] $gitrepo, [string] $tag)

    $url = "https://github.com/$gitrepo"

    [System.IO.Directory]::CreateDirectory("$opt/$dir")

    get_or_sync -dir "$dir/trunk" -url "$url.git"

    get_if_not_there -dir "$dir/$tag" -url "$url/archive/$tag.zip"
}

function main {
    prepare

    install_boost "1.64.0"
    install_boost "1.65.0"
    install_boost "1.66.0"
        
    get_or_sync -dir "libs/cmcstl2" -url "https://github.com/CaseyCarter/cmcstl2.git"
    get_or_sync -dir "libs/GSL" -url "https://github.com/Microsoft/GSL.git"
    get_or_sync -dir "libs/gsl-lite" -url "https://github.com/martinmoene/gsl-lite.git"
    get_or_sync -dir "libs/opencv" -url "https://github.com/opencv/opencv.git"
    get_or_sync -dir "libs/xtensor" -url "https://github.com/QuantStack/xtensor.git"
    get_or_sync -dir "libs/abseil" -url "https://github.com/abseil/abseil-cpp.git"
    get_or_sync -dir "libs/cctz" -url "https://github.com/google/cctz.git"
    get_or_sync -dir "libs/ctre" -url "https://github.com/hanickadot/compile-time-regular-expressions.git"
    get_or_sync -dir "libs/cppcoro" -url "https://github.com/lewissbaker/cppcoro.git"

    get_github_versioned_and_trunk -dir "libs/ulib" -gitrepo "stefanocasazza/ULib" -tag "v1.4.2"
    get_github_versioned_and_trunk -dir "libs/google-benchmark" -gitrepo "google/benchmark" -tag "v1.2.0"
    get_github_versioned_and_trunk -dir "libs/rangesv3" -gitrepo "ericniebler/range-v3" -tag "0.3.0"
    get_github_versioned_and_trunk -dir "libs/dlib" -gitrepo "davisking/dlib" -tag "v19.7"
    get_github_versioned_and_trunk -dir "libs/libguarded" -gitrepo "copperspice/libguarded" -tag "libguarded-1.1.0"
    get_github_versioned_and_trunk -dir "libs/brigand" -gitrepo "edouarda/brigand" -tag "1.3.0"

    get_if_not_there -dir "libs/eigen/v3.3.4" -url "http://bitbucket.org/eigen/eigen/get/3.3.4.zip"

}

main
