package hw6;

import java.util.Date;

abstract public class GeometricObject {
	
	private String color;
	private boolean filled;
	private Date dateCreated;
	
	protected GeometricObject() {
		dateCreated = new Date();
	}
	
	protected GeometricObject(String color, boolean filled) {
		this.color = color;
		this.filled = filled;
		dateCreated = new Date();
	}

	public String getColor() {
		return color;
	}

	public void setColor(String color) {
		this.color = color;
	}

	public boolean isFilled() {
		return filled;
	}

	public void setFilled(boolean filled) {
		this.filled = filled;
	}

	public Date getDateCreated() {
		return dateCreated;
	}
	
	public String toString() {
		return ("Color: " + color + "\nFilled: " + filled);
	}
	
	abstract public double getArea();
	abstract public double getPerimeter();

}
