#include "QtGame.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QtGame w;
	w.show();
	return a.exec();
}
