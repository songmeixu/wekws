FetchContent_Declare(portaudio
  URL      http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz
  URL_HASH SHA256=47efbf42c77c19a05d22e627d42873e991ec0c1357219c0d74ce6a2948cb2def
)
FetchContent_MakeAvailable(portaudio)
include_directories(${portaudio_SOURCE_DIR}/include)
