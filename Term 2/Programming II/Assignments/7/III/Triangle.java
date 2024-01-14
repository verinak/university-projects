package hw7;

public class Triangle {
	
	private double side1, side2, side3;
	private static int trianglesCreated = 0; 
	
	public Triangle() {
		side1 = side2 = side3 = 1;
	}
	
	public Triangle(double side1, double side2, double side3) throws IllegalTriangleException {
		if(side1 + side2 <= side3 || side2 + side3 <= side1 || side1 + side3 <= side2) {
			throw new IllegalTriangleException(side1, side2, side3);
		}
		else {
			this.side1 = side1;
			this.side2 = side2;
			this.side3 = side3;
			trianglesCreated ++;
		}
	}
	
	public String toString() {
		return ("Sides: " + side1 + "; " + side2 + "; " + side3);
	}
	
	public double getArea() {
		double s = getPerimeter()/2;
		return Math.sqrt(s*(s-side1)*(s-side2)*(s-side3));
	}

	public double getPerimeter() {
		return (side1 + side2 + side3);
	}

	public static int getTrianglesCreated() {
		return trianglesCreated;
	}

}
