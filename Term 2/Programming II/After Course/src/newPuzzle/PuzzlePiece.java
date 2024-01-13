package newPuzzle;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JPanel;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Rectangle;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

public class PuzzlePiece extends ImageIcon {
	static int piecesPlaced = 0;
	private Rectangle cornerRectangle;
	private Point finalLocation;
	private Point paintLocation;
	private boolean moveable;

	PuzzlePiece(String address, Rectangle cornerRectangle, Point finalLocation) {
		super(address);
		this.cornerRectangle = cornerRectangle;
		this.finalLocation = finalLocation;
		paintLocation = new Point((int)(Math.random()*280) + 620, (int)(Math.random()*400) + 20);
		moveable = true;
		
	}

	public Rectangle getCornerRectangle() {
		return cornerRectangle;
	}

	public boolean isMoveable() {
		return moveable;
	}

	public void setMoveable(boolean moveable) {
		this.moveable = moveable;
	}

	public Point getPaintLocation() {
		return paintLocation;
	}
	
	public void setPaintLocation(Point paintLocation) {
		this.paintLocation = paintLocation;
	}

	public Point getFinalLocation() {
		return finalLocation;
	}

	/*@Override
	public void mouseDragged(MouseEvent e) {
		// TODO Auto-generated method stub
		imageClicked = this;
		System.out.println("fml");
	}

	@Override
	public void mouseMoved(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}*/
}
