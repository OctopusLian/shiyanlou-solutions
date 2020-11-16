#ifndef stars_hpp
#define stars_hpp
#include <GL/glut.h>
class Star{
public:
    GLfloat radius;
    GLfloat selfSpeed,speed;
    GLfloat rgbaColor[4];
    GLfloat distance;

    Star* parentStar;

    Star(GLfloat radius,GLfloat distance,
    GLfloat selfSpeed,GLfloat speed,
    Star* parentStar);

    void drawStar();
    virtual void draw(){ drawStar(); }
    virtual void update(long timeSpan);
protected:
    GLfloat alphaSelf,alpha;
};
class Planet:public Star{
public:
    Planet(GLfloat radius,GLfloat distance,
    GLfloat speed,GLfloat selfSpeed,
    Star* parentStar,GLfloat rgbColor[3]);
    void drawPlanet();
    virtual void draw(){ drawPlanet(); drawStar(); }
};

class LightPlanet:public Planet {
public:
    LightPlanet(GLfloat Radius,GLfloat Distance,
    GLfloat Speed,GLfloat SelfSpeed,
    Star* parentStar,GLfloat rgbColor[]);
    void drawLight();
        virtual void draw() { drawLight(); drawPlanet(); drawStar();
}
};
Star::Star(GLfloat radius,GLfloat distance,
    GLfloat speed,GLfloat selfSpeed,
    Star* parent);
Planet::Planet(GLfloat radius,GLfloat distance,
    GLfloat speed,GLfloat selfSpeed,
    Star* parent,GLfloat rgbColor[3]);
LightPlanet::LightPlanet(GLfloat radius,GLfloat distance,
            GLfloat speed,GLfloat selfSpeed,
            Star* parent,GLfloat rgbColor[3]);
#endif