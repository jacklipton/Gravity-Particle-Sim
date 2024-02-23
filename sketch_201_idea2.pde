ArrayList<Particle> particles;

void setup() {
  size(800, 600);
  particles = new ArrayList<Particle>();
  background(0); // Set background to black
}

void draw() {
  // Adjust background inside draw loop to clear the screen and show particle motion
  background(0);
  
  // Update and render particles
  for (int i = particles.size() - 1; i >= 0; i--) {
    Particle p = particles.get(i);
    p.update();
    p.display();
    if (p.isDead()) {
      particles.remove(i);
    }
  }
}

void mouseMoved() {
  // Add new particles when mouse is moved
  particles.add(new Particle(new PVector(mouseX, mouseY)));
}

class Particle {
  PVector position;
  PVector velocity;
  PVector acceleration;
  float lifespan;

  Particle(PVector pos) {
    position = pos.copy();
    velocity = new PVector(random(-1, 1), random(-2, 2));//at generation give the particle a random velocity
    acceleration = new PVector(0, 0);
    lifespan = 255.0;
  }

  void update() {
    acceleration.mult(0); // Reset acceleration
    applyAttraction();
    velocity.add(acceleration);
    //velocity.limit(maxSpeed);
    position.add(velocity);
    position.add(velocity);
    lifespan -= 0.5;
    
  }
  //Replicating gravity
  void applyAttraction() {
    for (Particle other : particles) {
      if (other != this) {
        // Calculate force using Fg = M*m*G/r^2. In this case a general constant is used for the numerator
        //the goal is to adjust acceleration based on distance
        PVector force = PVector.sub(other.position, position);//gives vector defining distance and direction from one particle to another
        float distanceSq = force.magSq();
        distanceSq = constrain(distanceSq, 1, 1000); // Limit the distance to avoid extremely strong forces
        float strength = 0.1 * (1 / distanceSq); // Adjust the strength as desired
        force.setMag(strength);//scale the vector relative to the magintude of the force
        // Apply force
        acceleration.add(force);//add this force to the all the previously calculated forces //<>//

      }
    }
  }
  
  void display() {
    noStroke();
    fill(255, lifespan);
    ellipse(position.x, position.y, 8, 8); 
   
    // Draw a line indicating the direction of travel
    float arrowSize = constrain(sqrt(acceleration.magSq())*10000,1,40);
    print(arrowSize,"\n");
    pushMatrix();
    translate(position.x, position.y);
    rotate(acceleration.heading());
    stroke(255,lifespan);
    //create arrow
    line(0, 0, arrowSize, 0);
    line(arrowSize, 0, arrowSize - 5, 5);
    line(arrowSize, 0, arrowSize - 5, -5);
    popMatrix();
  }

  boolean isDead() {
    return (lifespan < 0.0);
  }
}
