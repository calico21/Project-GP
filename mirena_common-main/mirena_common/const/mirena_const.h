#ifndef mirena_const
#define mirena_const

// COMMON CONFIGURATION PARAMETERS FOR ALL NODES:

// THESE ARE NOT EMPIRICAL AND ARE JUST TEST VALUES!!!!!!!!!!!
//#define MR_CONST_MASS 200   // Mass of the vehicle
//#define MR_CONST_L_F 0.806    // Distance from center of mass to front axis
//#define MR_CONST_K_F 50000  // Front axis equivalent sideslip stiffness
//#define MR_CONST_L_R 1.8    // Distance from center of mass to rear axis
//#define MR_CONST_K_R 45000  // Rear axis equivalent sideslip stiffness
//#define MR_CONST_I_Z 122.0   // Yaw inertia of vehicle body

#define MR_CONST_MASS 1400   // Mass of the vehicle
#define MR_CONST_L_F 1    // Distance from center of mass to front axis
#define MR_CONST_K_F -12891  // Front axis equivalent sideslip stiffness
#define MR_CONST_L_R 1.8    // Distance from center of mass to rear axis
#define MR_CONST_K_R -8594  // Rear axis equivalent sideslip stiffness
#define MR_CONST_I_Z 3000   // Yaw inertia of vehicle body


//------------------------------[Vehicle Model]----------------------------------//
#define CAR_MASS 210 //Masa vehiculo
#define MAX_TRQ 90 //Max Nm en el motor 
#define I_ZZ 122.0 //Momento de inercia en eje Z (kg*M^4) (Modelo Juan Gastaminza)
#define T_WIDTH 1.185 //Track width REAR (m)
#define L_FRONT 0.806 //A Distance(from front axle to CDG) (m)
#define L_REAR 0.744 //B Distance (from CDG to rear axle)  (m)
#define H_CDG 0.27   //height of the CDG (en estatico)  (m)
#define GEAR_R 5.0     //Indice de Reducccion
#define R_WHEEL 0.2023 //Radio de la rueda (m)

//Actuator conversion ratios
#define APPS2ACC ((90.0/100.0)*GEAR_R *R_WHEEL)/CAR_MASS
#define BPPS2ACC (20/100)




#endif 
