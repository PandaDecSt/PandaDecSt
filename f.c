#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif




typedef struct {
    float re, im;
} Complex;


Complex cAdd(Complex x, Complex y)
{
    return (Complex){
        x.re + y.re, x.im + y.im,
    };
}

Complex cSub(Complex x, Complex y)
{
    return (Complex){
        x.re - y.re, x.im - y.im,
    };
}

Complex cMul(Complex x, Complex y)
{
    return (Complex){
        x.re * y.re - x.im * y.im,
        x.im * y.re + x.re * y.im,
    };
}




typedef struct {
    int fftLen;
    float adjC;
    int (*swpIdx)[2];
    Complex *vUni[2];
} FastDFTctx;


void fftCreateCtx(FastDFTctx *c, int N)
{
    int idx = 0, s = 0; c->fftLen = N;
    c->vUni[0] = malloc(sizeof(Complex) * N);
    c->swpIdx  = malloc(sizeof(int) * N);
    c->vUni[1] = c->vUni[0] + (N >> 1);
    c->adjC = (float)1 / c->fftLen;

    for (int i = 1; i < c->fftLen - 1; ++i)
    {
        s ^= N - N / ((i ^ (i - 1)) + 1);

        if (i < s)
        {
            c->swpIdx[idx][0] = i;
            c->swpIdx[idx][1] = s, ++idx;
        }
    }

    c->swpIdx[idx][0] = 0, s = N >> 1;
    float ang = M_PI * 2 / c->fftLen;
    Complex *vec_arr = c->vUni[0];

    for (int i = 0; i < s; ++i)
    {
        vec_arr[i | s].re = cosf(ang * i);
        vec_arr[i | s].im = sinf(ang * i);
        vec_arr[i].re =  vec_arr[i | s].re;
        vec_arr[i].im = -vec_arr[i | s].im;
    }
}

void fftDestroyCtx(FastDFTctx *c)
{
    free(c->swpIdx), free(c->vUni[0]);
}

void fftExecTrans(FastDFTctx *c, Complex *arr, int m)
{
    for (int (*sp)[2] = c->swpIdx; **sp; ++sp)
    {
        Complex s_tmp = arr[(*sp)[0]];
        arr[(*sp)[0]] = arr[(*sp)[1]];
        arr[(*sp)[1]] = s_tmp;
    }

    for (int sc = 2; sc <= c->fftLen; sc <<= 1)
    {
        int hs = sc >> 1, d = c->fftLen / sc;

        for (int t = 0; t < c->fftLen; t += sc)
            for (int k = 0; k < hs; ++k)
            {
                int u = k | t, v = u | hs;
                Complex x = arr[u], y = arr[v];
                y = cMul(y, c->vUni[m][k * d]);
                arr[u] = cAdd(x, y);
                arr[v] = cSub(x, y);
            }
    }

    for (int i = 0; m && i < c->fftLen; ++i)
        arr[i].re *= c->adjC, arr[i].im *= c->adjC;
}




typedef struct {
    float firstX;
    float *resu;
    float *tmp0;
    FastDFTctx FFT;
    Complex *base;
    Complex *tmp1;
} PolyITPctx;


float polyCalc(float x, float const *po, int N)
{
    float resu = po[0], px = 1;

    for (int i = 1; i < N; ++i)
        resu += po[i] * (px *= x);

    return resu;
}

void itpCreateCtx(PolyITPctx *c, int maxL)
{
    fftCreateCtx(&c->FFT, maxL);
    c->resu = malloc(sizeof(float)   * maxL * 2);
    c->base = malloc(sizeof(Complex) * maxL * 2);
    c->tmp0 = c->resu + maxL;
    c->tmp1 = c->base + maxL;
}

void itpDestroyCtx(PolyITPctx *c)
{
    fftDestroyCtx(&c->FFT);
    free(c->resu), free(c->base);
}

void itpSetClean(PolyITPctx *c)
{
    for (int i = 0; i < c->FFT.fftLen; ++i)
    {
        c->base[i].re = c->resu[i] = 0;
        c->base[i].im = c->tmp0[i] = 0;
    }

    c->base[0].re = 1;
}

static void itp__convBase(PolyITPctx *c, float x)
{
    for (int i = 0; i < c->FFT.fftLen; ++i)
    {
        c->tmp0[i] = c->base[i].re;
        c->tmp1[i].re = c->tmp1[i].im = 0;
    }

    c->tmp1[0].re = -x, c->tmp1[1].re = 1;
    fftExecTrans(&c->FFT, c->base, 0);
    fftExecTrans(&c->FFT, c->tmp1, 0);

    for (int i = 0; i < c->FFT.fftLen; ++i)
        c->base[i] = cMul(c->base[i], c->tmp1[i]);

    fftExecTrans(&c->FFT, c->base, 1);
}

void itpAddDeriv(PolyITPctx *c, float x, float d)
{
    float d0, d1, K;

    for (int i = 1; i < c->FFT.fftLen; ++i)
        c->tmp0[i - 1] = c->resu[i] * i;

    c->tmp0[c->FFT.fftLen - 1] = 0;
    d0 = polyCalc(x, c->tmp0, c->FFT.fftLen);

    for (int i = 1; i < c->FFT.fftLen; ++i)
        c->tmp0[i - 1] = c->base[i].re * i;

    c->tmp0[c->FFT.fftLen - 1] = 0;
    d1 = polyCalc(x, c->tmp0, c->FFT.fftLen);

    K = (d - d0) / d1, itp__convBase(c, x);

    for (int i = 0; i < c->FFT.fftLen; ++i)
        c->resu[i] += K * c->tmp0[i];
}

void itpAddSampl(PolyITPctx *c, float x, float y)
{
    float y0, y1, K;

    itp__convBase(c, x);
    y0 = polyCalc(x, c->resu, c->FFT.fftLen);
    y1 = polyCalc(x, c->tmp0, c->FFT.fftLen);
    K  = (y - y0) / y1;

    for (int i = 0; i < c->FFT.fftLen; ++i)
        c->resu[i] += K * c->tmp0[i];
}

float itpCalcResult(PolyITPctx *c, float x)
{
    return polyCalc(x, c->resu, c->FFT.fftLen);
}




#define FFT_LEN 8192
#define SCR_W   1024
#define SCR_H   192
#define COL_N   120

typedef struct {
    unsigned char r, g, b;
} PIX_t;

PIX_t SCR[SCR_W * SCR_H];


static void clearScr()
{
    for (int i = 0; i < SCR_W * SCR_H; ++i)
        SCR[i] = (PIX_t){10, 10, 10};
}

static PIX_t genPIX(int key)
{
    PIX_t p = {0, 0, 0};

    switch((key %= 1530) / 255)
    {
    case 0: p.r = 255, p.g = key;        break;
    case 1: p.g = 255, p.r = 510 - key;  break;
    case 2: p.g = 255, p.b = key - 510;  break;
    case 3: p.b = 255, p.g = 1020 - key; break;
    case 4: p.b = 255, p.r = key - 1020; break;
    case 5: p.r = 255, p.b = 1530 - key; break;
    }

    return p;
}

static void DrawCir(int px, int py, int r, PIX_t cl)
{
    int sqr = r * r, p = py * SCR_W + px;

    for (int x = 0; x <= r; ++x)
        for (int y = 0; y <= r; ++y)
            if (x + y <= r || x * x + y * y <= sqr)
            {
                int d = y * SCR_W;
                SCR[p + d + x] = SCR[p + d - x] = cl;
                SCR[p - d + x] = SCR[p - d - x] = cl;
            }
}

static void Draw(int x0, int y0, PIX_t col0, int x1, int y1, PIX_t col1)
{
    if (y0 < 0)
        y0 = 0;

    if (y1 < 0)
        y1 = 0;

    y1 = SCR_H - y1 - 5;
    y0 = SCR_H - y0 - 5;

    for (int y = SCR_H - 1; y > y1; --y)
    {
        SCR[y * SCR_W + x1 - 1] = col1;
        SCR[y * SCR_W + x1 + 1] = col1;
    }

    float dy = y1 - y0, dxi = 1.0f / (x1 - x0);

    for (int x = x0; x < x1; ++x)
    {
        float p = dxi * (x - x0);
        int y = y0 + p * dy;

        PIX_t c = (PIX_t){
            col1.r * p + col0.r * (1 - p),
            col1.g * p + col0.g * (1 - p),
            col1.b * p + col0.b * (1 - p),
        };

        SCR[y       * SCR_W + x] = c;
        SCR[(y - 1) * SCR_W + x] = c;
    }

    DrawCir(x1, y1, 4, col1);
}

static inline float winFunc(int i)
{
    return 1.0f - cosf(M_PI * i * 2 / FFT_LEN);
}

static inline float getFreqVal(Complex c)
{
    float e = log10f(c.re * c.re + c.im * c.im + 1e-5f);
    return e > 0? e : 0;
}


static const char* strf(const char *fmt, ...)
{
    static char buf[1024];
    va_list va;

    va_start(va, fmt);
    vsnprintf(buf, 1024, fmt, va);
    va_end(va);

    return buf;
}

static const char* basefn(const char *fn)
{
    static char buf[1024];
    int i = 0;

    for ( ; i < 1024 && fn[i]; ++i)
        buf[i] = fn[i];

    while (--i > 0 && fn[i] != '.'){ }

    return (buf[i] = 0, buf);
}

int main(int ac, char **av)
{
    if (ac != 2)
    {
        printf("使用方法：%s [音频文件]\n", *av);
        sleep(2);
        return 1;
    }

    FILE *vid = popen(strf("ffmpeg -y -hide_banner -f rawvideo "
                      "-pix_fmt rgb24 -r 60 -s 1024x192 -i - "
                      "-i \"%s\" -pix_fmt yuv420p \"%s.mp4\"",
                      av[1], basefn(av[1])), "wb");

    FILE *aud = popen(strf("ffmpeg -v error -i \"%s\" -ac 1"
                      " -ar 48000 -f f32le -", av[1]), "rb");

    Complex *arr  = malloc(sizeof(Complex)  * FFT_LEN);

    float *in_buf = malloc(sizeof(float) *
                        (FFT_LEN + COL_N * 4));

    int *map_freq = malloc(sizeof(int) *
                        (FFT_LEN / 2 + COL_N + 20));

    int *map_cnt= map_freq + FFT_LEN / 2, itpcnt = 0;
    int (*itprng)[2] = (int(*)[2])(map_cnt  + COL_N);
    float  (*flt)[4] = (float(*)[4])(in_buf + FFT_LEN);

    FastDFTctx FFT;
    PolyITPctx ITP;
    fftCreateCtx(&FFT, FFT_LEN);
    itpCreateCtx(&ITP, 16);


    for (int i = 0; i < COL_N; ++i)
        map_cnt[i] = 0;

    for (int i = 3; i < FFT_LEN / 2; ++i)
    {
        map_freq[i] = logf((float)i / 3) * COL_N * 0.138f;
        ++map_cnt[map_freq[i]];
    }

    for (int i = 0; i < COL_N - 1; ++i)
    {
        if (map_cnt[i] && !map_cnt[i + 1])
            itprng[itpcnt][0] = i;

        if (!map_cnt[i] && map_cnt[i + 1])
            itprng[itpcnt++][1] = i + 1;
    }


    int ret;

    do
    {
        for (int i = 0; i < FFT_LEN - 800; ++i)
            in_buf[i] = in_buf[i + 800];

        ret = fread(in_buf + FFT_LEN - 800, 4, 800, aud);

        for (int i = 0; i < FFT_LEN; ++i)
        {
            arr[i].re = in_buf[i] * winFunc(i);
            arr[i].im = 0;
        }

        fftExecTrans(&FFT, arr, 0);

        for (int i = 0; i < COL_N; ++i)
        {
            flt[i][0] = flt[i][1];
            flt[i][1] = flt[i][2];
            flt[i][2] = flt[i][3];
            flt[i][3] = 0;
        }

        for (int i = 3; i < FFT_LEN / 2; ++i)
            flt[map_freq[i]][3] += getFreqVal(arr[i]);

        for (int i = 0; i < COL_N; ++i)
            if (map_cnt[i] > 1)
                    flt[i][3] /= map_cnt[i];

        for (int i = 0; i < itpcnt; ++i)
        {
            int sta = itprng[i][0], d0 = itprng[i][1] - sta;
            int b0 = flt[itprng[i][0]] < flt[itprng[i][1]];
            
            itpSetClean(&ITP);
            itpAddSampl(&ITP, 0, flt[itprng[i][0]][3]);
            itpAddDeriv(&ITP, 0, 0);

            if (i < itpcnt - 1)
            {
                int b1 = flt[itprng[i + 1][0]] < flt[itprng[i + 1][1]];

                if (b0 == b1 && ++i)
                {
                    int d1 = d0 + itprng[i][1] - itprng[i][0];
                    itpAddSampl(&ITP, (float)d0 / d1, flt[itprng[i][0]][3]);
                    d0 = d1;
                }
            }

            itpAddSampl(&ITP, 1, flt[itprng[i][1]][3]);
            itpAddDeriv(&ITP, 1, 0);

            for (int p = 1; p <= d0; ++p)
                flt[sta + p][3] = itpCalcResult(&ITP, (float)p / d0);
        }

        clearScr();

        int _x = 0, _y = 0;
        PIX_t _col = (PIX_t){0, 0, 0};

        for (int i = 0; i < COL_N; ++i)
        {
            int y = (flt[i][0] + flt[i][1]
                  + flt[i][2] + flt[i][3] - 0.5f) * 3;

            int x = i * 8.3f + 10;
            PIX_t col = genPIX(y > 0? y << 3 : 0);

            Draw(_x, _y, _col, x, y, col);
            _x = x, _y = y, _col = col;
        }

        fwrite(SCR, 3 * SCR_W * SCR_H, 1, vid);

    } while (ret == 800);

    fftDestroyCtx(&FFT);
    itpDestroyCtx(&ITP);
    free(arr), free(in_buf), free(map_freq);
    return (pclose(vid), pclose(aud), 0);
}