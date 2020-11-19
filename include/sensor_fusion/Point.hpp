#ifndef POINT_H
#define POINT_H

class Point
{
private:
    double x, y;
public:
    Point(double x, double y)
    {
        this->x = x;
        this->y = y;
    }
    ~Point()
    {

    }

    double get_x()
    {
        return x;
    }

    double get_y()
    {
        return y;
    }
};

#endif