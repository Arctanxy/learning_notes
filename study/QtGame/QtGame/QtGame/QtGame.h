#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGame.h"

class QtGame : public QMainWindow
{
	Q_OBJECT

public:
	QtGame(QWidget *parent = Q_NULLPTR);

private:
	Ui::QtGameClass ui;
};
