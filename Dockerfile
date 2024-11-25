# Base image: Ruby with necessary dependencies for Jekyll
FROM ruby:3.2

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to configure bundler and Jekyll
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    BUNDLE_PATH=/usr/local/bundle

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy Gemfile and Gemfile.lock into the container
COPY Gemfile Gemfile.lock ./

# Install bundler and dependencies
RUN gem install bundler:2.3.26 && bundle install

# Copy the rest of the application into the container
COPY . .

# Expose port 4000 for the Jekyll server
EXPOSE 4000

# Command to serve the Jekyll site with live reload and auto-regeneration
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--watch"]
