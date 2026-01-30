//mand.frag
#version 450

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
        int findex, cindex;
        int maxIterations;
} pc;
int MAX_ITER = pc.maxIterations;
float burningShip();
float tricorn();
float mandOrbitTrap();
float multiBrot(double power);
float perpMand();
float celticMand();
float orbitAngle();
float boxTrap();
float ikeda();
float henon();
float lissajous();
float newFractal();

vec2 resolution = vec2(pc.width, pc.height);

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

float newton()
{
    //const int    MAX_ITER = 50;
    const double EPS     = 1e-6;

    dvec2 z = screenCoordD() * double(pc.zoom) +
              dvec2(pc.centerX, pc.centerY);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
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

float mand()
{
    //const int    MAX_ITER = 256;
    const double ESCAPE2  = 4.0;

    dvec2 p = screenCoordD();
    dvec2 center = dvec2(pc.centerX, pc.centerY);
    double zoom  = double(pc.zoom);
    dvec2 c = p * zoom + center;
    dvec2 z = dvec2(0.0);
    int i;
    for (i = 0; i < MAX_ITER; i++) {
        // z = z^2 + c
        z = dvec2(z.x*z.x-z.y*z.y, 2.0*z.x*z.y)+c;
        if (dot(z, z) > ESCAPE2) break;
    }
    if (i == MAX_ITER) return 0.0;

    double mag2 = dot(z, z);
    double mu = double(i) - double(log2(log2(float(mag2))));
    return float(clamp(mu / double(MAX_ITER), 0.0, 1.0));
}
/*
struct dd {
    double hi;
    double lo;
};

dd dd_from_parts(double hi, double lo) {
    dd r;
    r.hi = hi;
    r.lo = lo;
    return r;
}

dd dd_from_double(double a) {
    dd r;
    r.hi = a;
    r.lo = 0.0;
    return r;
}

double dd_to_double(dd a) {
    return a.hi + a.lo;
}

// Accurate two-sum for addition
dd dd_add(dd a, dd b) {
    double s = a.hi + b.hi;
    double v = s - a.hi;
    double t = ((b.hi - v) + (a.hi - (s - v))) + a.lo + b.lo;
    double hi = s + t;
    double lo = t - (hi - s);
    return dd_from_parts(hi, lo);
}

dd dd_sub(dd a, dd b) {
    // a - b = a + (-b)
    dd nb;
    nb.hi = -b.hi;
    nb.lo = -b.lo;
    return dd_add(a, nb);
}

// Multiplication using fma for best accuracy
dd dd_mul(dd a, dd b) {
    double p = a.hi * b.hi;
    // error of the hi*hi product + cross terms
    double err = fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi + a.lo * b.lo;
    double hi = p + err;
    double lo = err - (hi - p);
    return dd_from_parts(hi, lo);
}

dd dd_mul_double(dd a, double b) {
    // multiply dd by scalar double b
    double p = a.hi * b;
    double err = fma(a.hi, b, -p) + a.lo * b;
    double hi = p + err;
    double lo = err - (hi - p);
    return dd_from_parts(hi, lo);
}

// Square a dd value
dd dd_sqr(dd a) {
    double p = a.hi * a.hi;
    double err = fma(a.hi, a.hi, -p) + 2.0 * a.hi * a.lo + a.lo * a.lo;
    double hi = p + err;
    double lo = err - (hi - p);
    return dd_from_parts(hi, lo);
}

// dd complex (pair) helpers
struct dd2 {
    dd x;
    dd y;
};

dd2 dd2_from_dvec2(dvec2 v) {
    dd2 r;
    r.x = dd_from_double(v.x);
    r.y = dd_from_double(v.y);
    return r;
}

dd2 dd2_add(dd2 a, dd2 b) {
    dd2 r;
    r.x = dd_add(a.x, b.x);
    r.y = dd_add(a.y, b.y);
    return r;
}

dd2 dd2_mul_complex(dd2 a, dd2 b) {
    // complex multiply: (a.x + i a.y)*(b.x + i b.y)
    // real = a.x*b.x - a.y*b.y
    // imag = a.x*b.y + a.y*b.x
    dd axbx = dd_mul(a.x, b.x);
    dd ayby = dd_mul(a.y, b.y);
    dd real = dd_sub(axbx, ayby);

    dd axby = dd_mul(a.x, b.y);
    dd aybx = dd_mul(a.y, b.x);
    dd imag = dd_add(axby, aybx);

    dd2 r;
    r.x = real;
    r.y = imag;
    return r;
}

dd2 dd2_square(dd2 a) {
    // (x + i y)^2 = (x^2 - y^2) + i(2xy)
    dd x2 = dd_sqr(a.x);
    dd y2 = dd_sqr(a.y);
    dd real = dd_sub(x2, y2);

    dd xy = dd_mul(a.x, a.y);
    dd imag = dd_mul_double(xy, 2.0);

    dd2 r;
    r.x = real;
    r.y = imag;
    return r;
}

double dd2_dot_to_double(dd2 a) {
    // returns approximate double of dot(a,a) using hi+lo for each
    double ax = dd_to_double(a.x);
    double ay = dd_to_double(a.y);
    return ax*ax + ay*ay;
}

float doubleMand()
{
    const double ESCAPE2 = 4.0;

    // Build high-precision constants from push-constants
    dd zoom = dd_from_parts(pc.zoomhi, pc.zoomlo);
    dd centerX = dd_from_parts(pc.centerXhi, pc.centerXlo);
    dd centerY = dd_from_parts(pc.centerYhi, pc.centerYlo);

    // Map pixel -> complex plane at high precision
    dvec2 p = screenCoordD(); // returns dvec2
    // c = p * zoom + center (where p is double, zoom is dd)
    dd2 c;
    c.x = dd_add(dd_mul(dd_from_double(p.x), zoom), centerX);
    c.y = dd_add(dd_mul(dd_from_double(p.y), zoom), centerY);

    dd2 z;
    z.x = dd_from_double(0.0);
    z.y = dd_from_double(0.0);

    int maxIter = pc.maxIterations;
    int i;
    for (i = 0; i < maxIter; ++i) {
        // z = z^2 + c using dd2_square
        dd2 z2 = dd2_square(z);
        z = dd2_add(z2, c);

        // magnitude test: use combined double approximation for speed
        double mag2 = dd2_dot_to_double(z);
        if (mag2 > ESCAPE2) break;
    }

    if (i == maxIter)
        return 0.0;

    // Smooth iteration count:
    // compute final magnitude with better precision by converting dd parts
    double mag2_hi_lo = dd_to_double(dd_add(dd_sqr(z.x), dd_sqr(z.y))); // slightly more accurate
    // clamp and compute mu
    double mu = double(i) - double(log2(log2(float(max(mag2_hi_lo, 1e-300))))); // avoid log(0)
    return float(clamp(mu / double(maxIter), 0.0, 1.0));
}*/

float julia(dvec2 c)
{
    //const int    MAX_ITER = 256;
    const double ESCAPE2  = 4.0;

    dvec2 z = screenCoordD() * double(pc.zoom) +
              dvec2(pc.centerX, pc.centerY);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(z.x*z.x - z.y*z.y,
                  2.0*z.x*z.y) + c;
        if (dot(z, z) > ESCAPE2) break;
    }

    if (i == MAX_ITER) return 0.0;

    double mag2 = dot(z, z);
    double mu = double(i) - double(log2(log2(float(mag2))));
    return float(clamp(mu / double(MAX_ITER), 0.0, 1.0));
}

void main() //|||||||||||||||||||||||||||||MAIN|||||||||||||||||||||||||||||||||||||||||||||||||||||||||
{
    float v;

    int findex = pc.findex % 22;
    int cindex = pc.cindex % 5;

    switch(findex){
    //case -1: v = doubleMand(); break;
    case 0: v = mand(); break;
    case 1: v = julia(dvec2(0.3, 0.7)); break;
    case 2: v = burningShip(); break;
    case 3: v = tricorn(); break;
    case 4: v = mandOrbitTrap(); break;
    case 5: v = newton(); break;
    //case 6: v = multibrot(double(5)); break;
    case 7: v = perpMand(); break;
    case 8: v = celticMand(); break;
    case 9: v = orbitAngle(); break;
    case 10: v = boxTrap(); break;
    case 11: v = ikeda(); break;
    case 12: v = henon(); break;
    case 13: v = lissajous(); break;
    case 14: v = newFractal(); break;
    case 20: v = hash(screenCoord() * hash(vec2(pc.time,0.7))); break;
    default: outColor = vec4(1.0,0.0,0.0,1.0); return;
    }
    switch(cindex){
    case 0: outColor = vec4(vec3(v), 1.0); break; //grayscale
    case 1: outColor = vec4(brightnessToColor(v,0.0,1.0), 1.0); break; //custom
    case 2: outColor = vec4(hsv2rgb(vec3(v,1.0,1.0)), 1.0); break; //hsv
    case 3: outColor = vec4(colorDistance(v), 1.0); break; //trigonometric
    default: outColor = vec4(1.0,0.0,0.0,1.0); return;
    }
}
//||||||||||||||||||||||||||||||||||||END MAIN||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
float burningShip()
{
    //const int    MAX_ITER = 256;
    const double ESCAPE2  = 4.0;

    dvec2 p = screenCoordD();
    dvec2 c = p * double(pc.zoom) + dvec2(pc.centerX, pc.centerY);
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(abs(z.x), abs(z.y));
        z = dvec2(z.x*z.x - z.y*z.y,
                  2.0*z.x*z.y) + c;
        if (dot(z, z) > ESCAPE2) break;
    }

    if (i == MAX_ITER) return 0.0;

    double mu = double(i) - double(log2(log2(float(dot(z,z)))));
    return float(clamp(mu / double(MAX_ITER), 0.0, 1.0));
}

float tricorn()
{
    //const int    MAX_ITER = 256;
    const double ESCAPE2  = 4.0;

    dvec2 p = screenCoordD();
    dvec2 c = p * double(pc.zoom) + dvec2(pc.centerX, pc.centerY);
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(z.x*z.x - z.y*z.y,
                 -2.0*z.x*z.y) + c;
        if (dot(z, z) > ESCAPE2) break;
    }

    if (i == MAX_ITER) return 0.0;

    double mu = double(i) - double(log2(log2(float(dot(z,z)))));
    return float(clamp(mu / double(MAX_ITER), 0.0, 1.0));
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
    //const int MAX_ITER = 256;
    const double ESCAPE2 = 4.0;

    dvec2 c = screenCoordD() * double(pc.zoom) +
              dvec2(pc.centerX, pc.centerY);
    dvec2 z = dvec2(0.0);

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        z = dvec2(
            z.x*z.x - z.y*z.y + c.x,
            -2.0*z.x*z.y + c.y
        );
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
    dvec2 z = screenCoordD();
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
    dvec2 z = screenCoordD();
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
    vec2 p = screenCoord();
    float x = sin(3.0*p.x + pc.time);
    float y = sin(4.0*p.y);
    return 0.5 + 0.5*x*y;
}

float newFractal()
{
    // Placeholder for additional fractal implementations
    return 0.0;
}