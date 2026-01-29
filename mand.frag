//mand.frag
#version 450

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
    	float width, height;
    	float time;
        float sinTime, cosTime, tanTime;
	    double zoom;
    	double centerX, centerY;
        int findex, cindex;
        int maxIterations;
} pc;
//int MAX_ITER = pc.maxIterations;
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
    const int    MAX_ITER = 50;
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
    const int    MAX_ITER = 256;
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

float julia(dvec2 c)
{
    const int    MAX_ITER = 256;
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

    int findex = 0; //pc.findex % 22;
    int cindex = 1; //pc.cindex % 5;

    switch(findex){
    case 0: v = mand(); break;
    case 1: v = julia(dvec2(0.15, 0.7)); break;
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
    const int    MAX_ITER = 256;
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
    const int    MAX_ITER = 256;
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
    const int    MAX_ITER = 256;
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
    const int MAX_ITER = 256;
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
    const int MAX_ITER = 256;
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
    const int MAX_ITER = 256;
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
    const int MAX_ITER = 128;

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
    const int MAX_ITER = 256;

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