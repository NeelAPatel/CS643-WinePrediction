# Docker Image with Spark + Hadoop
FROM guoxiaojun2/spark-3.2.2-bin-hadoop3.2

# Setting working directory
WORKDIR /app

# Copy the local project files into the Docker container
COPY . .

# Switch to root user to install Python and pip
USER root

# Update the system and install Python
RUN apt-get update -y && apt-get install -y python3

# Install pip
RUN apt-get install -y python3-pip

# Install Python libraries
RUN pip3 install pyspark findspark boto3 numpy pandas scikit-learn datetime

# Change permissions for the run_scripts.sh
RUN chmod +x run_scripts.sh

# Switch back to the non-root user
USER ${SPARK_USER}

# Command to run on container start
CMD [ "./run_scripts.sh" ]
