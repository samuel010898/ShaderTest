//mand.frag
#version 450
#include "mand.frag.h"
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Push {
    	float width, height;
    	float time;
        float sinTime, cosTime, tanTime;
	    double zoom;
    	double centerX, centerY;
        //double zoomhi, zoomlo;
        //double centerXhi, centerXlo;
        //double centerYhi, centerYlo;
        int findex, cindex, sindex;
        int maxIterations;
        float scale;
} pc;
vec2 resolution = vec2(pc.width, pc.height);
int MAX_ITER = pc.maxIterations;

float mand(dvec2 c, dvec2 z);

vec2 pixelCoord()
{
    return gl_FragCoord.xy;
}

vec2 uv()
{
    return gl_FragCoord.xy / resolution; //0,1
}

vec2 screenCoord()//-1,1
{
    vec2 p = (gl_FragCoord.xy / resolution) * 2.0 - 1.0;
    float aspect = resolution.x / resolution.y;
    p.x *= aspect;
    return p;
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float hash1(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float valueNoise(vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash1(i);
    float b = hash1(i + vec2(1,0));
    float c = hash1(i + vec2(0,1));
    float d = hash1(i + vec2(1,1));

    vec2 u = f*f*(3.0 - 2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

vec2 curl(vec2 p)
{
    float e = 0.001;
    float n1 = valueNoise(p + vec2(0, e));
    float n2 = valueNoise(p - vec2(0, e));
    float n3 = valueNoise(p + vec2(e, 0));
    float n4 = valueNoise(p - vec2(e, 0));

    return vec2(n1 - n2, n4 - n3);
}

vec3 randColor(vec2 p) {
    return vec3(
        hash(p + 0.1),
        hash(p + 17.3),
        hash(p + 42.9)
    );
}

double remap(double t, double min1, double max1, double min2, double max2)
{
	if (min1 >= max1) {return min2;}
	return min2 + (max2 - min2) * ((t - min1) / (max1 - min1));
}

vec3 hsv2rgb(vec3 c)
{
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0/3.0, 1.0/3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 colorDistance(float v)
{
    return vec3(
        0.5 + 0.5*cos(3.0 + v*15.0),
        0.5 + 0.5*cos(1.0 + v*15.0),
        0.5 + 0.5*cos(5.0 + v*15.0)
    );
}

vec3 brightnessToColor(float v, float min, float max)
{
	if (max <= min) return vec3(1,0,0);
	if (v < min) return vec3(1.0,1.0,0.0); //yellow
	if (v > max) return vec3(0.0,1.0,1.0); //cyan
	float x = (v - min) / (max - min);
	float t = x * 7.0;
    	int seg = int(t);
    	float f = t - seg;
	//f = f * f * (3.0f - 2.0f * f); // smoothing
	float r=0, g=0, b=0;
    	switch (seg)
	{
        	case 0: b=(f); break; //black→blue
        	case 1: b=1.0; g=((f)); break; //blue→cyan
        	case 2: g=1.0; b=1.0-f; break; //cyan→green
        	case 3: g=1.0; r=((f)); break; //green→yellow
        	case 4: r=1.0; g=1.0-f; break; //yellow→red
        	case 5: r=1.0; b=((f)); break; //red→magenta
		case 6: r=1.0; g=((f)); b=1.0; break; //magenta→white
		default: return randColor(vec2(v,v-((720/2)/480/2)+720/2)); //overflow
    	}
    	return vec3(r, g, b);
}

dvec2 resolutionD()
{
    return dvec2(pc.width, pc.height);
}

dvec2 screenCoordD()
{
    dvec2 frag = dvec2(gl_FragCoord.xy);
    dvec2 res  = resolutionD();
    dvec2 p = (frag - res * 0.5) / res.y;
    return p;
}

dvec2 jitter()
{
    dvec2 j = dvec2(curl(vec2(screenCoordD().x * 2.0, screenCoordD().y * 2.0)).x, curl(vec2(screenCoordD().x * 2.0, screenCoordD().y * 2.0)).y);
    j = (j - 0.5);
    return j;
}

dvec2 windowCoordD()
{
    return screenCoordD()*double(pc.zoom)+dvec2(pc.centerX, pc.centerY);
}

float newton()
{
    int maxIter = int(ceil(float(MAX_ITER/5.0f)));
    const double EPS = 1e-6;

    dvec2 z = windowCoordD();

    int i;
    for (i = 0; i < maxIter; i++) {
        // f(z) = z^3 - 1
        dvec2 z2 = dvec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y);
        dvec2 z3 = dvec2(z2.x*z.x - z2.y*z.y,
                          z2.x*z.y + z2.y*z.x);

        dvec2 f  = z3 - dvec2(1.0, 0.0);

        // f'(z) = 3z^2
        dvec2 fp = 3.0 * z2;

        double denom = dot(fp, fp);
        if (denom < EPS) break;

        z -= dvec2(
            (f.x*fp.x + f.y*fp.y) / denom,
            (f.y*fp.x - f.x*fp.y) / denom
        );
    }

    return float(i) / float(MAX_ITER);
}

float remapLinear(float t, float minT, float maxT)
{
    if (maxT <= minT) return 0.0;
    return clamp((t - minT) / (maxT - minT), 0.0, 1.0);
}

float remapPower(float t, float minT, float maxT, float power)
{
    float x = remapLinear(t, minT, maxT);
    return pow(x, power);
}
//if (j > 0.0) c += jitter()*j; 
void main() //|||||||||||||||||||||||||||||MAIN|||||||||||||||||||||||||||||||||||||||||||||||||||||||||
{
    float v;

    int findex = pc.findex % 33;
    int cindex = pc.cindex % 12;
    int sindex = pc.sindex % 5;

    switch(findex){
    case 0: v = mand(windowCoordD(),vec2(0.0)); break;
    case 1: v = mand(dvec2(0.355, 0.355), windowCoordD()); break;
    case 2: v = mand(dvec2(-0.70176, -0.3842), windowCoordD()); break;
    case 3: v = burningShip(); break;
    case 4: v = tricorn(); break;
    case 5: v = mandOrbitTrap(); break;
    case 6: v = newton(); break;
    case 7: v = multibrot(double(/*pc.sinTime* */5.0)); break;
    case 8: v = perpMand(); break;
    case 9: v = celticMand(); break;
    case 10: v = orbitAngle(); break;
    case 11: v = boxTrap(); break;
    case 12: v = ikeda(); break;
    case 13: v = henon(); break;
    case 14: v = lissajous(); break;
    case 15: v = lyapunov(); break;
    case 16: v = collatz(); break;
    case 17: v = biomorph(); break;
    case 18: v = phoenix(); break;
    case 19: v = magnet1(); break;
    case 20: v = nova(); break;
    case 21: v = tetration(); break;
    case 22: v = gingerbreadman(); break;
    case 23: v = duffing(); break;
    case 24: v = tinkerbell(); break;
    case 25: v = gumowski_mira(); break;
    case 26: v = peter_de_jong(); break;
    case 27: v = clifford(); break;
    case 28: v = hopalong(); break;
    case 29: v = apollonian_gasket(); break;
    case 30: v = kleinian(); break;
    case 31: v = fractal_noise(); break;
    case 32: v = worley(); break;
    //case 99: v = hash(screenCoord() * hash(vec2(pc.time,0.7))); break;
    default: outColor = vec4(pc.sinTime,pc.cosTime*0.77,pc.tanTime*0.037,1.0); return;
    }

    switch(sindex % 5){
        case 0: v = remapLinear(v, 0.0, 1.0); break;
        case 1: v = remapPower(v, 0.0, 1.0, pc.scale); break;
        case 2: v = float(log(1.0 + v * (exp(pc.scale) - 1.0)) / log(exp(pc.scale))); break;
        case 3: v = float(exp(v * pc.scale) - 1.0) / (exp(pc.scale) - 1.0); break;
        case 4: v = sqrt(v); break;
        default: break;
    }

    switch(cindex){
    case 0: outColor = vec4(vec3(v), 1.0); break; //grayscale
    case 1: outColor = vec4(brightnessToColor(v,0.0,1.0), 1.0); break; //custom
    case 2: outColor = vec4(hsv2rgb(vec3(v,1.0,1.0)), 1.0); break; //hsv
    case 3: outColor = vec4(colorDistance(v), 1.0); break; //trigonometric
    case 4: outColor = vec4(viridis(v), 1.0); break;
    case 5: outColor = vec4(plasma(v), 1.0); break;
    case 6: outColor = vec4(magma(v), 1.0); break;
    case 7: outColor = vec4(hot(v), 1.0); break;
    case 8: outColor = vec4(coolwarm(v), 1.0); break;
    case 9: outColor = vec4(twilight(v), 1.0); break;
    case 10: outColor = vec4(binary(v), 1.0); break;
    case 11: outColor = vec4(histogramEqualized(v), 1.0); break;
    default: outColor = vec4(1.0,0.0,0.0,1.0); return;
    }
}
//||||||||||||||||||||||||||||||||||||END MAIN||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

float mand(dvec2 c, dvec2 z)
{
    const double ESCAPE2 = 16.0;
    const double logBail = log(float(ESCAPE2));

    //dvec2 c = windowCoordD();
    //dvec2 z = dvec2(0.0);

    int i = 0;
    for (; i < MAX_ITER; ++i)
    {
        z = dvec2(z.x*z.x-z.y*z.y, 2.0*z.x*z.y)+c; // z=z^2+c
        if (dot(z, z) > ESCAPE2) break;
    }

    if (i == MAX_ITER) return 0.0;

    double mag2 = dot(z, z);
    double mu = double(i) + 1.0 - log(log(float(mag2)) / float(logBail)) / log(2.0);
    float normalized = float( (mu + 1.0) / double(MAX_ITER + 4.0) );
    return clamp(normalized, 0.0, 1.0);
}

float burningShip()
{
    const double ESCAPE2  = 16.0;
    const double logBail = log(float(ESCAPE2));

    dvec2 c = windowCoordD();
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(abs(z.x), abs(z.y));
        z = dvec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y)+c;
        if (dot(z, z) > ESCAPE2) break;
    }

    if (i == MAX_ITER) return 0.0;

    double mag2 = dot(z, z);
    double mu = double(i) + 1.0 - log(log(float(mag2)) / float(logBail)) / log(2.0);
    float normalized = float( (mu + 1.0) / double(MAX_ITER + 4.0) );
    return clamp(normalized, 0.0, 1.0);
}

float tricorn()
{
    const double ESCAPE2  = 16.0;
    const double logBail = log(float(ESCAPE2));
    dvec2 c = windowCoordD();
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(z.x*z.x - z.y*z.y, -2.0*z.x*z.y) + c;
        if (dot(z, z) > ESCAPE2) break;
    }

    if (i == MAX_ITER) return 0.0;

    double mag2 = dot(z, z);
    double mu = double(i) + 1.0 - log(log(float(mag2)) / float(logBail)) / log(2.0);
    float normalized = float( (mu + 1.0) / double(MAX_ITER + 4.0) );
    return clamp(normalized, 0.0, 1.0);
}

float mandOrbitTrap()
{
    //const int    MAX_ITER = 256;
    const double ESCAPE2  = 4.0;

    dvec2 p = screenCoordD();
    dvec2 c = p * double(pc.zoom) + dvec2(pc.centerX, pc.centerY);
    dvec2 z = dvec2(0.0);

    double minDist = 1e9;

    for (int i = 0; i < MAX_ITER; i++) {
        z = dvec2(z.x*z.x - z.y*z.y,
                  2.0*z.x*z.y) + c;
        minDist = min(minDist, length(z));
        if (dot(z,z) > ESCAPE2) break;
    }

    return exp(float(-minDist * 5.0));
}

float multibrot(double power)
{
    //const int MAX_ITER = 256;
    const double ESCAPE2 = 4.0;

    dvec2 c = screenCoordD() * double(pc.zoom) +
              dvec2(pc.centerX, pc.centerY);
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        double r = length(z);
        double a = atan(float(z.y), float(z.x));
        r = pow(float(r), float(power));
        a *= power;
        z = dvec2(r*cos(float(a)), r*sin(float(a))) + c;
        if (dot(z,z) > ESCAPE2) break;
    }

    return float(i) / float(MAX_ITER);
}

float perpMand()
{
    const double ESCAPE2 = 16.0;

    dvec2 c = windowCoordD();
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(z.x*z.x-z.y*z.y+c.x,-2.0*z.x*z.y+c.y);
        if (dot(z,z) > ESCAPE2) break;
    }

    return float(i) / float(MAX_ITER);
}

float celticMand()
{
    //const int MAX_ITER = 256;
    const double ESCAPE2 = 4.0;

    dvec2 c = screenCoordD() * double(pc.zoom) +
              dvec2(pc.centerX, pc.centerY);
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(
            abs(z.x*z.x - z.y*z.y),
            2.0*z.x*z.y
        ) + c;
        if (dot(z,z) > ESCAPE2) break;
    }

    return float(i) / float(MAX_ITER);
}

float orbitAngle()
{
    //const int MAX_ITER = 128;

    dvec2 c = screenCoordD() * double(pc.zoom) +
              dvec2(pc.centerX, pc.centerY);
    dvec2 z = dvec2(0.0);

    double sum = 0.0;

    for (int i = 0; i < MAX_ITER; i++) {
        z = dvec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        sum += atan(float(z.y), float(z.x));
        if (dot(z,z) > 4.0) break;
    }

    return float(fract(sum * 0.1));
}

float boxTrap()
{
    //const int MAX_ITER = 256;

    dvec2 c = screenCoordD() * double(pc.zoom) +
              dvec2(pc.centerX, pc.centerY);
    dvec2 z = dvec2(0.0);

    double dmin = 1e9;

    for (int i = 0; i < MAX_ITER; i++) {
        z = dvec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        dmin = min(dmin, max(abs(z.x), abs(z.y)));
        if (dot(z,z) > 4.0) break;
    }

    return exp(float(-dmin * 4.0));
}

float ikeda()
{
    dvec2 z = windowCoordD();
    for (int i = 0; i < 20; i++) {
        double t = 0.4 - 6.0 / (1.0 + dot(z,z));
        z = dvec2(
            1.0 + z.x*cos(float(t)) - z.y*sin(float(t)),
            z.x*sin(float(t)) + z.y*cos(float(t))
        );
    }
    return float(fract(length(z)));
}

float henon()
{
    dvec2 z = windowCoordD();
    for (int i = 0; i < 20; i++) {
        z = dvec2(
            1.0 - 1.4*z.x*z.x + z.y,
            0.3*z.x
        );
    }
    return float(fract(length(z)));
}

float lissajous()
{
    dvec2 p = windowCoordD();
    double x = double(sin(float(3.0*p.x + pc.time)));
    double y = double(sin(float(4.0*p.y)));
    return float(0.5 + 0.5*x*y);
}

float lyapunov() {
dvec2 p = windowCoordD();
double a = p.x * 1.5 + 2.5; // Map x to [2.5, 4.0]
double b = p.y * 1.5 + 2.5; // Map y to [2.5, 4.0]
const int transient = 1000;
const int measured = 1000;
const double clip = 1e10; // To prevent overflow in log
double x = 0.5;
double sum = 0.0;
for (int i = 0; i < transient + measured; i++) {
double r = (i % 2 == 0) ? a : b;
x = r * x * (1.0 - x);
if (i >= transient) {
double deriv = r * (1.0 - 2.0 * x);
sum += log(float(abs(deriv) + 1e-10)); // Avoid log(0)
}
}
double lambda = sum / double(measured);
lambda = clamp(lambda, -clip, clip); // Prevent extremes
return float(remap(lambda, -5.0, 5.0, 0.0, 1.0)); // Negative stable (0), positive chaotic (1); adjust range based on typical values for visual contrast
}
float collatz() {
dvec2 p = windowCoordD();
dvec2 z = p; // Complex starting point
const double bailout = 1e-3;
const double max_mag = 1e6; // Escape if too large
int i;
for (i = 0; i < MAX_ITER; i++) {
// Smoothed complex Collatz: z -> (1/4)(2 + 7z - (2 + 5z)cos(πz)) or similar; here's a basic variant
double arg = 3.1415926535 * z.x; // Approximate pi * Re(z)
double cos_pi_z = cos(float(arg)) * cosh(float(3.1415926535 * z.y)); // Real part approx
double sin_pi_z = sin(float(arg)) * sinh(float(3.1415926535 * z.y)); // Imag part approx
dvec2 term1 = 0.25 * (dvec2(2.0, 0.0) + 7.0 * z);
dvec2 term2 = 0.25 * (dvec2(2.0, 0.0) + 5.0 * z) * dvec2(cos_pi_z, -sin_pi_z); // Complex cos(πz)
z = term1 - term2;
double mag2 = dot(z, z);
if (mag2 < bailout * bailout) break; // Converged to cycle near 1
if (mag2 > max_mag * max_mag) break; // Escaped
}
if (i == MAX_ITER) return 0.0;
double mu = double(i) - log(log(float(sqrt(dot(z, z))))) / log(2.0);
return float(clamp(mu / double(MAX_ITER), 0.0, 1.0));
}
float biomorph() {
dvec2 c = windowCoordD();
dvec2 z = dvec2(0.0);
const double escape_re = 5.0;
const double escape_im = 5.0;
int i;
for (i = 0; i < MAX_ITER; i++) {
z = dvec2(z.x*z.x-z.y*z.y, 2.0*z.x*z.y)+c;
if (abs(z.x) > escape_re || abs(z.y) > escape_im) break;
}
if (i == MAX_ITER) return 0.0;
double mu = double(i) - log(log(float(max(abs(z.x), abs(z.y))))) / log(2.0);
return float(clamp(mu / double(MAX_ITER), 0.0, 1.0));
}
float phoenix() {
dvec2 p = windowCoordD();
dvec2 c = p * double(pc.zoom) + dvec2(pc.centerX, pc.centerY);
dvec2 z = dvec2(0.0);
dvec2 prev_z = dvec2(0.0);
dvec2 param = dvec2(0.5667, -0.5); // Fixed p for phoenix shape
const double ESCAPE2 = 4.0;
int i;
for (i = 0; i < MAX_ITER; i++) {
dvec2 z_new = dvec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c + param * prev_z;
prev_z = z;
z = z_new;
if (dot(z, z) > ESCAPE2) break;
}
if (i == MAX_ITER) return 0.0;
double mu = double(i) - log2(log2(float(dot(z, z))));
return float(clamp(mu / double(MAX_ITER), 0.0, 1.0));
}
float magnet1() {
dvec2 p = windowCoordD();
dvec2 c = p * double(pc.zoom) + dvec2(pc.centerX, pc.centerY);
dvec2 z = dvec2(0.0);
const double ESCAPE2 = 4.0;
int i;
for (i = 0; i < MAX_ITER; i++) {
// z = ((z^2 + c - 1) / (2z + c - 2))^2
dvec2 num = dvec2(z.x * z.x - z.y * z.y + c.x - 1.0, 2.0 * z.x * z.y + c.y);
dvec2 den = dvec2(2.0 * z.x + c.x - 2.0, 2.0 * z.y + c.y);
double den_mag2 = dot(den, den);
if (den_mag2 < 1e-10) break; // Avoid division by zero
dvec2 div = dvec2((num.x * den.x + num.y * den.y) / den_mag2, (num.y * den.x - num.x * den.y) / den_mag2);
z = dvec2(div.x * div.x - div.y * div.y, 2.0 * div.x * div.y);
if (dot(z, z) > ESCAPE2) break;
}
if (i == MAX_ITER) return 0.0;
double mu = double(i) - log2(log2(float(dot(z, z))));
return float(clamp(mu / double(MAX_ITER), 0.0, 1.0));
}
float nova() {
dvec2 z = windowCoordD();
//dvec2 z = screenCoordD(); // z starts as pixel
dvec2 c = dvec2(1.0, 0.0); // For z^3 - 1 = 0, adjust c for other polys
const double relaxation = 1.0;
const double EPS = 1e-5;
int i;
for (i = 0; i < MAX_ITER; i++) {
// For f(z) = z^3 + c*z - 1, but simplify to z^3 - 1
dvec2 z2 = dvec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
dvec2 f = dvec2(z2.x * z.x - z2.y * z.y, z2.x * z.y + z2.y * z.x) - dvec2(1.0, 0.0) + c * z;
dvec2 fp = 3.0 * z2 + c;
double den = dot(fp, fp);
if (den < EPS) break;
dvec2 dz = dvec2((f.x * fp.x + f.y * fp.y) / den, (f.y * fp.x - f.x * fp.y) / den);
z -= relaxation * dz;
if (dot(dz, dz) < EPS * EPS) break;
}
return float(i) / float(MAX_ITER); // Color by iterations to converge
}
float tetration() {
dvec2 p = windowCoordD();
dvec2 z = p; // Complex base
const double conv_radius = 0.065988; // e^{-e} approx
const double EPS = 1e-6;
int max_height = 5; // Finite tetration height
dvec2 w = dvec2(1.0, 0.0); // Start with w=1
int i;
for (i = 0; i < max_height; i++) {
// w = z^w
double r = length(w);
double theta = atan(float(w.y), float(w.x));
double log_r = log(float(r));
double new_r = exp(float(log_r * z.x - theta * z.y));
double new_theta = log_r * z.y + theta * z.x;
w = dvec2(new_r * cos(float(new_theta)), new_r * sin(float(new_theta)));
if (length(w - z) < EPS) break; // Converged for infinite, but here finite
}
double mag = length(w);
return float(clamp(mag / (conv_radius * 10.0), 0.0, 1.0)); // Normalize; tetration grows fast, so arbitrary scaling
}
float gingerbreadman() {
dvec2 p = windowCoordD();
dvec2 z = p; // Start from pixel as initial
double min_dist = 1e9;
int iters = MAX_ITER; // Fixed for attractor approximation
for (int i = 0; i < iters; i++) {
z = dvec2(1.0 - z.y + abs(z.x), z.x);
min_dist = min(min_dist, dot(z, z)); // Trap to origin or something
}
return exp(float(-sqrt(min_dist) * 5.0)); // Similar to orbit trap
}
float duffing() {
dvec2 p = windowCoordD();
double a = p.x * 2.0; // Param a around [-2,2]
double b = p.y * 0.5; // Param b small
dvec2 z = dvec2(0.0, 0.1); // Fixed initial
const double ESCAPE2 = 100.0;
int i;
for (i = 0; i < MAX_ITER; i++) {
dvec2 z_new;
z_new.x = z.y;
z_new.y = -b * z.x + a * z.y - z.y * z.y * z.y;
z = z_new;
if (dot(z, z) > ESCAPE2) break;
}
return float(i) / float(MAX_ITER);
}
float tinkerbell() {
dvec2 p = windowCoordD();
dvec2 z = p; // Initial from pixel
double a = 0.9;
double b = -0.6013;
double c_ = 2.0;
double d = 0.5;
const double ESCAPE2 = 4.0;
int i;
for (i = 0; i < MAX_ITER; i++) {
dvec2 z_new;
z_new.x = z.x * z.x - z.y * z.y + a * z.x + b * z.y;
z_new.y = 2.0 * z.x * z.y + c_ * z.x + d * z.y;
z = z_new;
if (dot(z, z) > ESCAPE2) break;
}
return float(i) / float(MAX_ITER);
}
float gumowski_mira() {
dvec2 p = windowCoordD();
dvec2 z = p; // Initial
double a = -0.008;
double b = 0.05;
double c_ = 2.0;
double min_dist = 1e9;
const int iters = 100;
for (int i = 0; i < iters; i++) {
double f_x = c_ * z.x + 2.0 * (1.0 - c_) * z.x * z.x / (1.0 + z.x * z.x);
dvec2 z_new;
z_new.x = z.y + a * (1.0 - b * z.y * z.y) * z.y + f_x;
double f_new = c_ * z_new.x + 2.0 * (1.0 - c_) * z_new.x * z_new.x / (1.0 + z_new.x * z_new.x);
z_new.y = -z.x + f_new;
z = z_new;
min_dist = min(min_dist, dot(z, z));
}
return exp(float(-sqrt(min_dist) * 5.0));
}
float peter_de_jong() {
dvec2 p = windowCoordD();
dvec2 z = dvec2(0.0);
double a = p.x * 3.0; // Param space for a, etc.
double b = p.y * 3.0;
double c_ = 2.4;
double d = -2.1;
double density = 0.0;
const int iters = 1000; // More for density
for (int i = 0; i < iters; i++) {
dvec2 z_new;
z_new.x = sin(float(a * z.y)) - cos(float(b * z.x));
z_new.y = sin(float(c_ * z.x)) - cos(float(d * z.y));
z = z_new;
// Approximate density by how close to pixel, but since per pixel, use fract(length)
if (i > 10) density += exp(float(-dot(z - p, z - p) * 10.0)); // Point trap to current p, but hacky
}
return float(clamp(density / float(iters - 10), 0.0, 1.0));
}
float clifford() {
dvec2 p = windowCoordD();
dvec2 z = dvec2(0.0);
double a = p.x * 2.0;
double b = p.y * 2.0;
double c_ = 1.5;
double d = 1.5;
const int iters = 100;
double min_dist = 1e9;
for (int i = 0; i < iters; i++) {
dvec2 z_new;
z_new.x = sin(float(a * z.y)) + c_ * cos(float(a * z.x));
z_new.y = sin(float(b * z.x)) + d * cos(float(b * z.y));
z = z_new;
min_dist = min(min_dist, dot(z, z));
}
return exp(float(-sqrt(min_dist) * 5.0));
}
float hopalong() {
dvec2 p = windowCoordD();
dvec2 z = dvec2(0.0);
double a = p.x * 10.0;
double b = p.y * 10.0;
double c_ = 1.0;
const int iters = 100;
double min_dist = 1e9;
for (int i = 0; i < iters; i++) {
dvec2 z_new;
double sign_x = z.x >= 0.0 ? 1.0 : -1.0;
z_new.x = z.y - sign_x * sqrt(abs(b * z.x - c_));
z_new.y = a - z.x;
z = z_new;
min_dist = min(min_dist, dot(z, z));
}
return exp(float(-sqrt(min_dist) * 5.0));
}

float apollonian_gasket() {
    dvec2 p = windowCoordD();

    // generator circles (one bounding circle + two inner generators).
    const int DEPTH = max(8, max(4, MAX_ITER / 16)); // ensure some minimum depth
    dvec2 centers[3];
    centers[0] = dvec2(0.0, 0.0);   // bounding
    centers[1] = dvec2(0.5, 0.0);   // inner
    centers[2] = dvec2(-0.25, 0.433); // inner (approx 60°)

    double radii[3];
    radii[0] = 1.0;   // bounding radius
    radii[1] = 0.5;   // inner radius (tweak to get better tangency)
    radii[2] = 0.5;

    int iter = 0;

    // iterate: if point is outside one of the generator circles invert it
    // in that circle. This produces the nested inversions that form the gasket.
    for (int i = 0; i < DEPTH; ++i) {
        bool didInvert = false;
        for (int j = 0; j < 3; ++j) {
            dvec2 d = p - centers[j];
            double dist2 = dot(d, d);
            double r2 = radii[j] * radii[j];

            // If outside the circle, invert into it.
            if (dist2 > r2 && dist2 > 0.0) {
                double k = r2 / dist2;
                p = centers[j] + k * d;
                didInvert = true;
                iter++;
                break; // invert once per outer loop iteration (typical approach)
            }
        }
        if (!didInvert) {
            // point already lies inside one of the generator circles
            break;
        }
    }

    return float(iter) / float(DEPTH);
}

float kleinian() {
dvec2 p = windowCoordD();
const int max_iters = 20;
// Simple Schottky group with two generators
dvec2 a = dvec2(0.0, 1.0); // Circle centers/radii placeholders
double ra = 0.5;
dvec2 b = dvec2(0.0, -1.0);
double rb = 0.5;
int i;
for (i = 0; i < max_iters; i++) {
// Möbius transform approx; alternate inversions
double da = length(p - a);
if (da < ra) {
p = a + (ra * ra / (da * da)) * (p - a);
} else {
double db = length(p - b);
if (db < rb) {
p = b + (rb * rb / (db * db)) * (p - b);
}
}
if (length(p) > 2.0) break; // Escape
}
return float(i) / float(max_iters);
}

float fractal_noise() {
dvec2 p = windowCoordD() * 5.0; // Scale for visibility
const int octaves = 8;
float amp = 0.5;
float freq = 1.0;
float sum = 0.0;
float max_sum = 0.0;
for (int i = 0; i < octaves; i++) {
sum += amp * valueNoise(vec2(p.x * freq, p.y * freq));
max_sum += amp;
amp *= 0.5;
freq *= 2.0;
}
return sum / max_sum;
}

float worley() {
dvec2 p = windowCoordD() * 5.0;
const int num_points = 9; // 3x3 grid
dvec2 i = floor(vec2(p.x, p.y));
float min_dist = 1e9;
for (int y = -1; y <= 1; y++) {
for (int x = -1; x <= 1; x++) {
dvec2 neighbor = i + dvec2(double(x), double(y));
dvec2 point = neighbor + dvec2(hash1(vec2(neighbor)), hash1(vec2(neighbor + 0.1)));
float dist = length(vec2(p.x - point.x, p.y - point.y));
min_dist = min(min_dist, dist);
}
}
return clamp(min_dist, 0.0, 1.0);
}

vec3 viridis(float t) {
    const vec3 c0 = vec3(0.267, 0.004, 0.329);
    const vec3 c1 = vec3(0.282, 0.100, 0.616);
    const vec3 c2 = vec3(0.133, 0.533, 0.898);
    const vec3 c3 = vec3(0.120, 0.812, 0.747);
    const vec3 c4 = vec3(0.666, 0.902, 0.364);
    const vec3 c5 = vec3(0.992, 0.906, 0.145);
    t = clamp(t, 0.0, 1.0);
    if (t < 0.2) return mix(c0, c1, t / 0.2);
    else if (t < 0.4) return mix(c1, c2, (t - 0.2) / 0.2);
    else if (t < 0.6) return mix(c2, c3, (t - 0.4) / 0.2);
    else if (t < 0.8) return mix(c3, c4, (t - 0.6) / 0.2);
    else return mix(c4, c5, (t - 0.8) / 0.2);
}

vec3 plasma(float t) {
    const vec3 c0 = vec3(0.050, 0.029, 0.527);
    const vec3 c1 = vec3(0.366, 0.032, 0.652);
    const vec3 c2 = vec3(0.698, 0.153, 0.416);
    const vec3 c3 = vec3(0.938, 0.418, 0.182);
    const vec3 c4 = vec3(0.988, 0.891, 0.561);
    t = clamp(t, 0.0, 1.0);
    float s = t * 4.0;
    int i = int(s);
    float f = fract(s);
    if (i == 0) return mix(c0, c1, f);
    else if (i == 1) return mix(c1, c2, f);
    else if (i == 2) return mix(c2, c3, f);
    else return mix(c3, c4, f);
}

vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 r = vec3(0.001462, 0.000466, 0.013866);
    vec3 g = vec3(1.66023, -0.845108, 0.179594);
    vec3 b = vec3(-5.33275, 6.13342, -1.81635);
    return r + g * t + b * t * t;
}

vec3 hot(float t) {
    t = clamp(t, 0.0, 1.0);
    if (t < 0.333) return vec3(3.0 * t, 0.0, 0.0);
    else if (t < 0.666) return vec3(1.0, 3.0 * (t - 0.333), 0.0);
    else return vec3(1.0, 1.0, 3.0 * (t - 0.666));
}

vec3 coolwarm(float t) {
    t = clamp(t, 0.0, 1.0);
    if (t < 0.5) return mix(vec3(0.23, 0.299, 0.754), vec3(0.865, 0.865, 0.865), t * 2.0);
    else return mix(vec3(0.865, 0.865, 0.865), vec3(0.706, 0.016, 0.150), (t - 0.5) * 2.0);
}

vec3 twilight(float t) {
    t = fract(t);
    return 0.5 + 0.5 * cos(6.28318 * (vec3(0.95, 0.7, 0.6) + t * vec3(1.0, 1.0, 1.0)));
}

vec3 binary(float t) {
    return vec3(step(0.5, t));
}

vec3 histogramEqualized(float t) {
    return vec3(pow(t, 1.0 / pc.scale));
}