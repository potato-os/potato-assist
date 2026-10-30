# Potato Assist app image: three llama.cpp-family engines built for the Raspberry Pi 5 (Cortex-A76), plus the
# controller. CI builds it natively on arm64 runners; it can also be built on any arm64 Docker host with
#   docker build -t potato-assist potato_assist
# The engine stages build in parallel; BUILD_JOBS is per stage.
ARG BUILD_JOBS=2
ARG CPU_FLAGS="-mcpu=cortex-a76"

# ---- engine 1: ik_llama.cpp (Bonsai 2 ternary models, PQ2_0 / Q2_0)
FROM debian:trixie AS ik
ARG BUILD_JOBS
ARG CPU_FLAGS
# main 2026-10-23; includes the Bonsai Q2_0/PQ2_0 output fix (3d7b18cf0).
ARG IK_LLAMA_SHA=804ee29c28a5e892df6e84941cb78628ed4c266c
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init ik \
    && git -C ik remote add origin https://github.com/ikawrakow/ik_llama.cpp.git \
    && git -C ik fetch --depth 1 origin "${IK_LLAMA_SHA}" \
    && git -C ik checkout --detach FETCH_HEAD
# Self-speculative decoding (ngram/suffix) with the vision projector loaded, for text-only contexts.
COPY ik-patches /src/ik-patches
RUN git -C ik apply /src/ik-patches/*.patch
# Potato OS ik profile: KleidiAI off, IQK FA for all quants, LTO; static, no BLAS.
RUN cmake -S ik -B build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF -DGGML_LTO=ON -DGGML_OPENMP=ON \
        -DGGML_CPU_KLEIDIAI=OFF -DGGML_IQK_FA_ALL_QUANTS=ON -DLLAMA_BUILD_TESTS=OFF \
        "-DCMAKE_C_FLAGS=-fno-strict-aliasing ${CPU_FLAGS}" \
        "-DCMAKE_CXX_FLAGS=-fno-strict-aliasing ${CPU_FLAGS}" \
    && cmake --build build --config Release -j "${BUILD_JOBS}" --target llama-server

# ---- engine 2: millie-runtime + the Potato OS patch series (Millie 1.1, hot experts on the CPU)
FROM debian:trixie AS millie
ARG BUILD_JOBS
ARG CPU_FLAGS
ARG MILLIE_SHA=8b55b878dece729827039460741340c46c2bf153
ARG MILLIE_ARCHIVE_SHA256=892300a1603e79806c2dbb1f945dd9d137af4a3ba5ee0646cc400850c9305e93
ARG PATCH_SERIES_SHA256=520266146a39409ee7e26dab032be9e30762ae8f43513b1d1ab429e466b958ba
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake curl ca-certificates patch \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY millie-patches /src/patches
# On top of the series: LoRA adapters with hot experts (non-expert tensors only).
COPY millie-local-patches /src/local-patches
RUN curl -fsSL -o millie.tar.gz "https://github.com/llmsforall/millie-runtime/archive/${MILLIE_SHA}.tar.gz" \
    && echo "${MILLIE_ARCHIVE_SHA256}  millie.tar.gz" | sha256sum -c - \
    && [ "$(cat patches/*.patch | sha256sum | cut -d' ' -f1)" = "${PATCH_SERIES_SHA256}" ] \
    && tar -xzf millie.tar.gz && mv "millie-runtime-${MILLIE_SHA}" millie \
    && for p in patches/*.patch local-patches/*.patch; do patch -p1 --forward --batch --silent -d millie -i "/src/$p"; done \
    # GCC spells function-level arch extensions "+i8mm"; the series was written for clang (Apple, NDK).
    && grep -rl 'target("i8mm")' millie/ggml/src | xargs -r sed -i 's/target("i8mm")/target("+i8mm")/g'
RUN cmake -S millie -B build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF -DGGML_LTO=ON -DGGML_OPENMP=ON -DGGML_CPU_KLEIDIAI=OFF \
        -DLLAMA_HOT_CPU=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
        "-DCMAKE_C_FLAGS=${CPU_FLAGS}" "-DCMAKE_CXX_FLAGS=${CPU_FLAGS}" \
    && cmake --build build --config Release -j "${BUILD_JOBS}" --target llama-server

# ---- engine 3: mainline llama.cpp (LFM3 / LFM3-VL)
FROM debian:trixie AS llama
ARG BUILD_JOBS
ARG CPU_FLAGS
# master 2026-10-29; LFM3-1.7B/LFM3-VL-2B (#31252) and LFM3 video input (#31283) are merged.
ARG LLAMA_CPP_SHA=eb4e244be77a906a183cd4a00fd51d9ad3200955
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init llama \
    && git -C llama remote add origin https://github.com/ggml-org/llama.cpp.git \
    && git -C llama fetch --depth 1 origin "${LLAMA_CPP_SHA}" \
    && git -C llama checkout --detach FETCH_HEAD
# Static, no BLAS, no HTTPS (models arrive through the controller); Q4_0 is repacked for the A76 at load.
RUN cmake -S llama -B build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF -DGGML_LTO=ON -DGGML_OPENMP=ON -DGGML_CPU_KLEIDIAI=OFF \
        -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF \
        "-DCMAKE_C_FLAGS=${CPU_FLAGS}" "-DCMAKE_CXX_FLAGS=${CPU_FLAGS}" \
    && cmake --build build --config Release -j "${BUILD_JOBS}" --target llama-server

# ---- the app
FROM debian:trixie-slim
ARG CPU_FLAGS
ARG IK_LLAMA_SHA=804ee29c28a5e892df6e84941cb78628ed4c266c
ARG MILLIE_SHA=8b55b878dece729827039460741340c46c2bf153
ARG LLAMA_CPP_SHA=eb4e244be77a906a183cd4a00fd51d9ad3200955
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates jq python3 python3-aiohttp python3-yaml \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ik /src/build/bin/llama-server /opt/engines/ik/
COPY --from=millie /src/build/bin/llama-server /opt/engines/millie/
COPY --from=llama /src/build/bin/llama-server /opt/engines/llama/
COPY www /www
COPY controller.py ollama_api.py grouped_context.py json_schema_to_grammar.py run.sh /
RUN chmod 755 /run.sh /opt/engines/*/* \
    && echo "CPU_FLAGS=${CPU_FLAGS} IK_LLAMA_SHA=${IK_LLAMA_SHA} MILLIE_SHA=${MILLIE_SHA} LLAMA_CPP_SHA=${LLAMA_CPP_SHA}" \
        > /etc/potato-assist-build

CMD ["/run.sh"]
